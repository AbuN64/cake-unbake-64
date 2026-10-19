"""Memory accesses read out of extracted assembly: which function loads or stores which offset of which object.

An object is named by a base: a global symbol, a function argument register at entry, a pointer loaded from another
base's field, or an indexed element address of another base. Registers are followed by a forward dataflow over each
function alone (moves, `addiu` offsets, `%hi`/`%lo` pairs, loaded pointers; a register holding different values on two
paths is dropped), never across a call: a call clobbers the caller-saved registers, and what a callee does with a
pointer is not followed. The result is evidence, not a proof: an access through a register the flow lost is not seen.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Base = tuple[Any, ...]  # ("sym", name) | ("arg", function, index) | ("load", Base, offset) | ("idx", Base, offset)

_LINE = re.compile(r"^\s*/\*\s*[0-9A-Fa-f]+\s+([0-9A-Fa-f]{8})\s+[0-9A-Fa-f]{8}\s*\*/\s+([a-z][\w.]*)\s*(.*?)\s*$")
_LABEL = re.compile(r"^\s*(\.?[A-Za-z_][\w.]*):\s*$")
_MEM = re.compile(r"^(.*)\((\$\w+)\)$")
_RELOC = re.compile(r"^%(hi|lo)\(\s*([A-Za-z_][\w.]*)\s*(?:([+-])\s*(0x[0-9A-Fa-f]+|\d+))?\s*\)$")

# opcode -> (width, kind, is_store); kind b/h/w/d are integers whose signedness the opcode may or may not name
MEMORY = {
    "lb": (1, "s8", False), "lbu": (1, "u8", False), "sb": (1, "i8", True),
    "lh": (2, "s16", False), "lhu": (2, "u16", False), "sh": (2, "i16", True),
    "lw": (4, "i32", False), "sw": (4, "i32", True), "lwu": (4, "i32", False),
    "lwc1": (4, "f32", False), "swc1": (4, "f32", True),
    "ld": (8, "i64", False), "sd": (8, "i64", True),
    "ldc1": (8, "f64", False), "sdc1": (8, "f64", True),
}  # fmt: skip
_UNALIGNED = frozenset({"lwl", "lwr", "swl", "swr", "ldl", "ldr", "sdl", "sdr", "ll", "sc", "lld", "scd"})
_BRANCH = frozenset({
    "beq", "bne", "beqz", "bnez", "bgez", "bgtz", "blez", "bltz", "bgezal", "bltzal", "b", "bal",
    "bc1t", "bc1f", "beql", "bnel", "beqzl", "bnezl", "bgezl", "bgtzl", "blezl", "bltzl", "bc1tl", "bc1fl",
})  # fmt: skip
_ALWAYS = frozenset({"b", "bal"})
_NO_DEST = frozenset({
    "sb", "sh", "sw", "sd", "swc1", "sdc1", "swl", "swr", "sdl", "sdr", "j", "jr", "nop", "syscall", "break", "cache",
    "sync", "mult", "multu", "div", "divu", "dmult", "dmultu", "ddiv", "ddivu", "mthi", "mtlo", "mtc0", "dmtc0",
    "mtc1", "dmtc1", "ctc1", "ctc0", "teq", "tne", "tge", "tlt", "tgeu", "tltu", "jr.hb", "eret",
})  # fmt: skip
_CALLER_SAVED = (
    "$at",
    "$v0",
    "$v1",
    "$a0",
    "$a1",
    "$a2",
    "$a3",
    "$t0",
    "$t1",
    "$t2",
    "$t3",
    "$t4",
    "$t5",
    "$t6",
    "$t7",
    "$t8",
    "$t9",
    "$ra",
    "$k0",
    "$k1",
)
_ARGS = ("$a0", "$a1", "$a2", "$a3")
_ALIAS = {"$fp": "$s8", "$r0": "$zero", "$0": "$zero"}


@dataclass(frozen=True)
class Val:
    """What a register holds: `kind` 'ref' is the address base+add, 'hi' a `lui` of that address's %hi, 'src' a value
    just loaded from an access (so a later dereference proves that field is a pointer)."""

    kind: str
    base: Base | None = None
    add: int = 0
    origin: tuple[Base, int] | None = None  # the (base, offset) field a loaded value came from


@dataclass(frozen=True)
class Access:
    function: str
    pc: int  # vram of the instruction
    base: Base
    offset: int
    width: int
    kind: str  # s8 u8 i8 s16 u16 i16 i32 f32 i64 f64 (i = signedness not named by the opcode)
    store: bool


@dataclass(frozen=True)
class Pointer:
    """A field that is a pointer: dereferenced, called, passed to a call, or given a symbol address."""

    base: Base
    offset: int
    why: str  # deref | call | fnptr | stored
    callee: str | None = None  # why == "call": the function the value was passed to, at argument `arg`
    arg: int = 0


@dataclass
class Function:
    name: str
    vram: int
    accesses: list[Access]
    pointers: list[Pointer]
    calls: list[str]


def _reg(token: str) -> str:
    token = token.strip()
    return _ALIAS.get(token, token)


def _is_gpr(token: str) -> bool:
    return token.startswith("$") and not re.match(r"^\$f\d", token)


def _number(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None


def _reloc(text: str) -> tuple[str, str, int] | None:
    found = _RELOC.match(text.strip())
    if not found:
        return None
    sign = -1 if found.group(3) == "-" else 1
    return found.group(1), found.group(2), sign * (_number(found.group(4)) or 0) if found.group(4) else 0


def _split(operands: str) -> list[str]:
    return [part.strip() for part in re.split(r",(?![^()]*\))", operands)] if operands else []


@dataclass
class _Ins:
    pc: int
    op: str
    args: list[str]
    label: str | None = None  # the label that sits before this instruction


def _parse(lines: Iterable[str]) -> Iterator[tuple[str, list[_Ins]]]:
    """(function name, instructions) for each glabel/endlabel pair of one assembly file's text."""
    name: str | None = None
    code: list[_Ins] = []
    pending: list[str] = []
    for line in lines:
        text = line.rstrip()
        if text.startswith("glabel "):
            name, code, pending = text.split()[1], [], []
        elif text.startswith("endlabel "):
            if name is not None:
                yield name, code
            name = None
        elif name is not None:
            found = _LINE.match(text)
            if found:
                code.append(
                    _Ins(
                        int(found.group(1), 16),
                        found.group(2),
                        _split(found.group(3)),
                        pending[-1] if pending else None,
                    )
                )
                pending = []
                continue
            label = _LABEL.match(text)
            if label:
                pending.append(label.group(1))
                for earlier in pending[:-1]:  # several labels in a row all point at the next instruction
                    code.append(_Ins(-1, "label", [earlier]))
    return


def _memory(args: list[str]) -> tuple[str, str] | None:
    """(offset text, base register) of an `imm(reg)` operand."""
    if len(args) < 2:
        return None
    found = _MEM.match(args[1])
    return (found.group(1).strip() or "0", _reg(found.group(2))) if found else None


class _Flow:
    def __init__(self, name: str, code: list[_Ins]) -> None:
        self.name = name
        self.code = [c for c in code if c.op != "label"]
        self.labels: dict[str, int] = {}
        index = 0
        for c in code:
            if c.op == "label":
                self.labels[c.args[0]] = index
            else:
                if c.label:
                    self.labels[c.label] = index
                index += 1

    def targets(self, i: int) -> list[int]:
        """Instruction indexes control reaches after instruction i, with the delay slot of a branch before them."""
        ins = self.code[i]
        previous = self.code[i - 1] if i > 0 else None
        if previous is not None and (previous.op in _BRANCH or previous.op in ("j", "jr", "jal", "jalr")):
            out: list[int] = []
            if previous.op in _BRANCH:
                label = previous.args[-1] if previous.args else ""
                if label in self.labels:
                    out.append(self.labels[label])
                if previous.op not in _ALWAYS and i + 1 < len(self.code):
                    out.append(i + 1)
            elif previous.op == "j" and previous.args and previous.args[0] in self.labels:
                out.append(self.labels[previous.args[0]])
            elif previous.op in ("jal", "jalr") and i + 1 < len(self.code):
                out.append(i + 1)
            return out
        del ins
        return [i + 1] if i + 1 < len(self.code) else []

    def step(self, i: int, state: dict[str, Val], sink: list[Access | Pointer] | None) -> dict[str, Val]:
        """The register state after instruction i; with a sink, also the accesses and pointer proofs it makes."""
        ins = self.code[i]
        op, args = ins.op, ins.args
        out = dict(state)

        def get(token: str) -> Val | None:
            token = _reg(token)
            if token == "$zero":
                return Val("zero")
            return state.get(token)

        def deref(value: Val | None, why: str) -> None:
            if sink is not None and value is not None and value.origin is not None:
                sink.append(Pointer(value.origin[0], value.origin[1], why))

        if op in MEMORY or op in _UNALIGNED:
            mem = _memory(args)
            width, kind, store = MEMORY.get(op, (0, "", op.startswith("s")))
            if mem is not None:
                offset_text, base_reg = mem
                held = get(base_reg)
                relocation = _reloc(offset_text)
                target: tuple[Base, int] | None = None
                if relocation and relocation[0] == "lo" and held is not None and held.kind == "hi":
                    target = (("sym", relocation[1]), relocation[2])
                elif relocation is None and held is not None and held.kind == "ref" and held.base is not None:
                    number = _number(offset_text)
                    if number is not None:
                        target = (held.base, held.add + number)
                if held is not None and held.kind == "ref":
                    deref(held, "deref")
                if target is not None and sink is not None and width and target[0] != ("stack",):
                    sink.append(Access(self.name, ins.pc, target[0], target[1], width, kind, store))
                if store and target is not None and width == 4 and sink is not None:
                    stored = get(args[0])
                    if stored is not None and stored.kind == "ref" and stored.base and stored.base[0] == "sym":
                        sink.append(Pointer(target[0], target[1], "stored"))
                if not store and _is_gpr(args[0]):
                    dest = _reg(args[0])
                    if op == "lw" and target is not None and target[0] != ("stack",):
                        out[dest] = Val("ref", ("load", target[0], target[1]), 0, target)
                    else:
                        out.pop(dest, None)
            elif not store and args and _is_gpr(args[0]):
                out.pop(_reg(args[0]), None)
            return out
        if op in ("jal", "jalr", "jr", "j"):
            if op in ("jalr", "jr") and args and _reg(args[-1]) != "$ra":
                held = get(args[-1])
                if sink is not None and held is not None and held.origin is not None:
                    sink.append(Pointer(held.origin[0], held.origin[1], "fnptr"))
            return out  # a call completes after its delay slot: see _after
        if op in _BRANCH or op in _NO_DEST:
            return out
        if not args or not _is_gpr(args[0]):
            return out
        dest = _reg(args[0])
        if dest == "$zero":
            return out
        value: Val | None = None
        if op == "lui" and len(args) == 2:
            relocation = _reloc(args[1])
            if relocation and relocation[0] == "hi":
                value = Val("hi", ("sym", relocation[1]), relocation[2])
        elif op in ("addiu", "addi", "daddiu") and len(args) == 3:
            source = get(args[1])
            relocation = _reloc(args[2])
            if _reg(args[1]) == "$sp":
                value = Val("ref", ("stack",), 0)
            elif relocation and relocation[0] == "lo" and source is not None and source.kind == "hi":
                value = Val("ref", ("sym", relocation[1]), relocation[2])
            elif source is not None and source.kind == "ref" and source.base is not None and relocation is None:
                number = _number(args[2])
                if number is not None:
                    value = Val("ref", source.base, source.add + number, source.origin)
        elif op in ("addu", "daddu", "or", "add") and len(args) == 3:
            left, right = get(args[1]), get(args[2])
            if right is not None and right.kind == "zero":
                value = _copy(left)
            elif left is not None and left.kind == "zero":
                value = _copy(right)
            elif op != "or":
                value = _indexed(left, right)
        elif op in ("move",) and len(args) == 2:
            value = _copy(get(args[1]))
        if value is None:
            out.pop(dest, None)
        else:
            out[dest] = value
        return out

    def run(self) -> Function:
        n = len(self.code)
        entering: list[dict[str, Val] | None] = [None] * n
        entry = {reg: Val("ref", ("arg", self.name, k), 0) for k, reg in enumerate(_ARGS)}
        if n == 0:
            return Function(self.name, 0, [], [], [])
        entering[0] = entry
        work = [0]
        labeled = sorted(set(self.labels.values()))
        budget = 400 * n  # the joins only widen, so this never ends a real function early; it bounds a lattice slip
        while True:
            while work and budget > 0:
                budget -= 1
                i = work.pop()
                state = entering[i]
                if state is None:
                    continue
                after = self._after(i, state, None)
                for j in self.targets(i):
                    known = entering[j]
                    merged = after if known is None else _meet(known, after)
                    if merged != entering[j]:
                        entering[j] = merged
                        work.append(j)
            unseen = [j for j in labeled if j < n and entering[j] is None]  # jump-table targets: no known predecessor
            if not unseen or budget <= 0:
                break
            entering[unseen[0]] = {}
            work = [unseen[0]]
        accesses: list[Access | Pointer] = []
        for i in range(n):
            state = entering[i]
            if state is not None:
                self._after(i, state, accesses)
        calls = [c.args[0] for c in self.code if c.op == "jal" and c.args]
        return Function(
            self.name,
            self.code[0].pc,
            [a for a in accesses if isinstance(a, Access)],
            [p for p in accesses if isinstance(p, Pointer)],
            calls,
        )

    def _after(self, i: int, state: dict[str, Val], sink: list[Access | Pointer] | None) -> dict[str, Val]:
        out = self.step(i, state, sink)
        previous = self.code[i - 1] if i > 0 else None
        if previous is not None and previous.op in ("jal", "jalr"):  # the call completes after its delay slot
            if sink is not None and previous.op == "jal" and previous.args:
                for k, reg in enumerate(_ARGS):
                    held = out.get(reg)
                    if held is not None and held.origin is not None:
                        sink.append(Pointer(held.origin[0], held.origin[1], "call", previous.args[0], k))
            for reg in _CALLER_SAVED:
                out.pop(reg, None)
        return out


def _copy(value: Val | None) -> Val | None:
    return value if value is not None and value.kind in ("ref", "hi") else None


def _indexed(left: Val | None, right: Val | None) -> Val | None:
    """base + index: the element address of an array at base, whose stride this flow does not know."""
    for one, other in ((left, right), (right, left)):
        if one is not None and one.kind == "ref" and one.base is not None and (other is None or other.kind != "ref"):
            return Val("ref", ("idx", one.base, one.add), 0, one.origin)
    return None


def _element(value: Val) -> tuple[Base, int] | None:
    """(array base, its first element's offset) when the value is a ref to an element of an array or to the array."""
    if value.kind != "ref" or value.base is None:
        return None
    if value.base[0] == "idx":
        return value.base[1], value.base[2]
    return value.base, value.add


def _join(a: Val, b: Val | None) -> Val | None:
    """What a register holds where two paths meet. Addresses into one array that differ (a loop's induction pointer
    walking it) join to an indexed base at the lowest offset seen, so every path agrees on one base."""
    if a == b:
        return a
    if b is None or a.kind != "ref" or b.kind != "ref" or a.base is None or b.base is None:
        return None
    origin = a.origin if a.origin == b.origin else None
    if a.base == b.base:
        if a.base[0] == "idx":  # a stride walking an element address: no offset within the element is known
            return Val("ref", a.base, 0, origin)
        return Val("ref", ("idx", a.base, min(a.add, b.add)), 0, origin)
    first, second = _element(a), _element(b)
    if first is not None and second is not None and first[0] == second[0] and "idx" in (a.base[0], b.base[0]):
        return Val("ref", ("idx", first[0], min(first[1], second[1])), 0, origin)
    return None


def _meet(a: dict[str, Val], b: dict[str, Val]) -> dict[str, Val]:
    out: dict[str, Val] = {}
    for reg, value in a.items():
        joined = _join(value, b.get(reg))
        if joined is not None:
            out[reg] = joined
    return out


def functions(text: str) -> list[Function]:
    """Every function (glabel ... endlabel) of one assembly file, with its accesses."""
    return [_Flow(name, code).run() for name, code in _parse(text.splitlines())]


def read(path: Path) -> list[Function]:
    return functions(path.read_text(errors="replace"))
