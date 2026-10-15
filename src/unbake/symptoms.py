"""Pure byte comparisons and the closed vocabulary of measured symptoms."""

import re
from collections import Counter
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher, unified_diff
from itertools import zip_longest
from typing import Any

import rabbitizer

from unbake.contracts import Json

_BOOLS = frozenset(
    (
        "bytes_differ",
        "unproved",
        "no_measurement",
        "compile_failed",
        "unresolved",
        "register_dominant",
        "sp_offset_only",
        "missing_struct_member",
    )
)
_COUNTS = frozenset(
    ("repeated_deleted_runs", "float_register_differences", "literal_immediates_off_four", "plateau_probes")
)
_DELTAS = frozenset(("size_delta_bytes", "frame_delta_bytes"))
_INTEGER = r"[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)"


def _lines(data: bytes) -> list[tuple[str, list[str]]]:
    if len(data) % 4:
        return []
    result = []
    for i in range(len(data) // 4):
        text = rabbitizer.Instruction(int.from_bytes(data[4 * i : 4 * i + 4], "big"), vram=4 * i).disassemble()
        parts = text.split(None, 1)
        result.append((parts[0], [s.strip() for s in parts[1].split(",")] if len(parts) > 1 else []))
    return result


def _integer(token: str) -> int:
    return int(token, 16 if "x" in token.lower() else 10)


def _kind(b: tuple[str, list[str]], t: tuple[str, list[str]]) -> str:
    if b[0] != t[0] or len(b[1]) != len(t[1]):
        return ""
    pairs = [(x, y) for x, y in zip(b[1], t[1], strict=True) if x != y]
    tokens = [token for pair in pairs for token in pair]
    if tokens and all(re.fullmatch(r"\$[\w]+", s) and s != "$sp" and not re.fullmatch(r"\$f\d+", s) for s in tokens):
        return "register"
    if tokens and all(re.fullmatch(r"\$f\d+", s) for s in tokens):
        return "float"
    if tokens and all(re.fullmatch(_INTEGER + r"\(\$sp\)", s) for s in tokens):
        return "sp"
    if len(pairs) == 1 and all(re.fullmatch(_INTEGER, s) for s in tokens):
        delta = abs(_integer(tokens[0]) - _integer(tokens[1]))
        if 0 < delta <= 16 and delta % 4 == 0:
            return "off-four"
    return ""


def _frame(lines: list[tuple[str, list[str]]]) -> int | None:
    for mnemonic, operands in lines:
        if (
            mnemonic == "addiu"
            and len(operands) == 3
            and operands[:2] == ["$sp", "$sp"]
            and re.fullmatch(_INTEGER, operands[2])
            and _integer(operands[2]) < 0
        ):
            return -_integer(operands[2])
    return None


def measure(built: bytes, target: bytes, section: str) -> Json:
    facts: dict[str, bool | int] = {"bytes_differ": built != target}
    if len(built) != len(target):
        facts["size_delta_bytes"] = len(built) - len(target)
    if section != ".text" or len(built) % 4 or len(target) % 4:
        return facts
    b, t = _lines(built), _lines(target)
    ops = SequenceMatcher(None, [m for m, _ in b], [m for m, _ in t], autojunk=False).get_opcodes()
    lengths = Counter(i2 - i1 for op, i1, i2, _, _ in ops if op == "delete")
    repeated = sum(count for count in lengths.values() if count >= 2)
    if repeated >= 2:
        facts["repeated_deleted_runs"] = repeated
    pairs = [
        (x, y)
        for op, i1, i2, j1, j2 in ops
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1)
        for x, y in zip(b[i1:i2], t[j1:j2], strict=True)
        if x != y
    ]
    kinds = Counter(_kind(x, y) for x, y in pairs)
    if pairs:
        facts["register_dominant"] = kinds["register"] / len(pairs) >= 0.8
        facts["sp_offset_only"] = kinds["sp"] == len(pairs)
    for kind, key in (("float", "float_register_differences"), ("off-four", "literal_immediates_off_four")):
        if kinds[kind]:
            facts[key] = kinds[kind]
    bf, tf = _frame(b), _frame(t)
    if bf is not None and tf is not None and bf != tf:
        facts["frame_delta_bytes"] = tf - bf
    return facts


def score(built: bytes, target: bytes) -> float:
    if built == target:
        return 1.0
    if not target:
        return 0.0
    step = 4 if len(built) % 4 == len(target) % 4 == 0 else 1
    b, t = ([data[i : i + step] for i in range(0, len(data), step)] for data in (built, target))
    matched = sum(block.size for block in SequenceMatcher(None, b, t, autojunk=False).get_matching_blocks())
    return max(0.0, min(matched / len(t), 1.0 - 2**-53))


def literals(
    built: bytes, target: bytes, read_built: Callable[[int, int], bytes], read_target: Callable[[int, int], bytes]
) -> bytes:
    """The built code with each float load's low half set to the target's where both addresses hold the same literal,
    so a compiler literal the linker placed elsewhere in the unit's rodata is not a difference."""
    highs: tuple[dict[int, int], dict[int, int]] = ({}, {})
    out = bytearray(built)
    for i in range(0, min(len(built), len(target)) - 3, 4):
        words = tuple(int.from_bytes(d[i : i + 4], "big") for d in (built, target))
        for word, high in zip(words, highs, strict=True):
            if word >> 26 == 0x0F:  # lui
                high[(word >> 16) & 31] = (word & 0xFFFF) << 16
        b, t = words
        if b != t and b >> 16 == t >> 16 and b >> 26 in (0x31, 0x35):  # lwc1, ldc1
            size = 8 if b >> 26 == 0x35 else 4
            at = [
                h.get((w >> 21) & 31, 0) + (w & 0xFFFF) - ((w & 0x8000) << 1) for w, h in zip(words, highs, strict=True)
            ]
            if (value := read_built(at[0], size)) and value == read_target(at[1], size):
                out[i + 2 : i + 4] = target[i + 2 : i + 4]
    return bytes(out)


def _blind(line: tuple[str, list[str]]) -> str:
    return line[0] + " " + re.sub(r"\$\w+", "$", ", ".join(line[1]))


def _common(x: Sequence[Any], y: Sequence[Any]) -> SequenceMatcher[Any]:
    return SequenceMatcher(None, x, y, autojunk=False)


def shift(built: bytes, target: bytes) -> Json:
    """Diagnostic only, never a landing or fuzzy score: where words first differ and how many align, blind and exact."""
    words = [[data[i : i + 4] for i in range(0, len(data) - len(data) % 4, 4)] for data in (built, target)]
    b, t = _lines(built), _lines(target)
    first = next((i for i, (x, y) in enumerate(zip_longest(*words)) if x != y), None)

    def text(lines: list[Any], i: int | None) -> str | None:
        return f"{lines[i][0]} {', '.join(lines[i][1])}".strip() if i is not None and i < len(lines) else None

    def aligned(x: Sequence[Any], y: Sequence[Any]) -> int:
        return sum(m.size for m in _common(x, y).get_matching_blocks())

    blind = [[_blind(x) for x in lines] for lines in (b, t)]
    where = None if first is None else {"index": first, "built": text(b, first), "target": text(t, first)}
    return {
        "first_difference": where,
        "aligned": {"exact": aligned(*words), "register_blind": aligned(*blind)},
        "positional": sum(x == y for x, y in zip(*words, strict=False)),
    }


def register_votes(built: bytes, target: bytes) -> dict[str, Counter[str]]:
    """Per register the built code names, the registers the target names in the same operand of aligned instructions."""
    b, t = _lines(built), _lines(target)
    matcher = _common([_blind(x) for x in b], [_blind(x) for x in t])
    votes: dict[str, Counter[str]] = {}
    for block in matcher.get_matching_blocks():
        for x, y in zip(b[block.a : block.a + block.size], t[block.b : block.b + block.size], strict=True):
            for have, want in zip(*(re.findall(r"\$\w+", ",".join(z[1])) for z in (x, y)), strict=True):
                votes.setdefault(have, Counter())[want] += 1
    return votes


def from_missing(missing: Sequence[str]) -> Json:
    facts = {}
    for gap in missing:
        match = re.search(r"size (\d+) != (\d+)", gap)
        if match:
            facts["size_delta_bytes"] = int(match[1]) - int(match[2])
        for key, token in (
            ("bytes_differ", "bytes differ"),
            ("unproved", "unproved"),
            ("no_measurement", "no measurement"),
            ("compile_failed", "compile.error"),
            ("unresolved", "unresolved"),
        ):
            if token in gap:
                facts[key] = True
        if re.search(r"has no member|incomplete type", gap):
            facts["missing_struct_member"] = True
    return facts


def merge(facts: Sequence[Json]) -> Json:
    result = {}
    for fact in facts:
        for key, value in fact.items():
            if key in _BOOLS:
                if value:
                    result[key] = True
            elif key in _COUNTS:
                result[key] = result.get(key, 0) + value
            elif key in _DELTAS:
                result.setdefault(key, value)
            elif key == "score":
                result[key] = min(result.get(key, value), value)
    return {key: value for key, value in result.items() if key not in _COUNTS or value != 0}


class _Line(str):
    """A diff line that compares equal by its key, so addresses never cause differences."""

    key: str

    def __new__(cls, text: str, key: str) -> "_Line":
        line = super().__new__(cls, text)
        line.key = key
        return line

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Line) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


def _aligned(data: bytes, vram: int) -> list[_Line]:
    lines = _lines(data)
    offsets = [
        rabbitizer.Instruction(int.from_bytes(data[4 * i : 4 * i + 4], "big"), vram=4 * i) for i in range(len(lines))
    ]
    plain = [f"{m} {', '.join(o[:-1] if w.isBranch() else o)}" for (m, o), w in zip(lines, offsets, strict=True)]
    result = []
    for n, word in enumerate(offsets):
        target = n + word.getBranchOffset() // 4 if word.isBranch() else None
        landing = plain[target] if target is not None and 0 <= target < len(plain) else "outside"
        key = plain[n] if target is None else f"{plain[n]} -> [{landing}]"
        shown = plain[n] if target is None else f"{plain[n]} -> {vram + 4 * target:08X}"
        result.append(_Line(f"{vram + 4 * n:08X}: {shown}", key))
    return result


def listing(data: bytes, vram: int) -> list[str]:
    """The instructions of `data` as `ADDRESS: text` lines, the way the diff shows them."""
    return [str(line) for line in _aligned(data, vram)]


def diff(built: bytes, target: bytes, vram: int) -> str:
    return "\n".join(unified_diff(_aligned(target, vram), _aligned(built, vram), "target", "built", lineterm="", n=3))
