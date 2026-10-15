"""Pure symptom measurements using hand-assembled big-endian MIPS words."""

import pytest

from unbake import symptoms


def _words(*words: int) -> bytes:
    return b"".join(word.to_bytes(4, "big") for word in words)


def _addu(rd: int = 2, rs: int = 4, rt: int = 5) -> int:
    return (rs << 21) | (rt << 16) | (rd << 11) | 0x21


def _addiu(rt: int, rs: int, immediate: int) -> int:
    return 0x24000000 | (rs << 21) | (rt << 16) | (immediate & 0xFFFF)


def _load(rt: int, base: int, offset: int) -> int:
    return 0x8C000000 | (base << 21) | (rt << 16) | (offset & 0xFFFF)


def _float(fd: int, fs: int = 4, ft: int = 6) -> int:
    return 0x46000000 | (ft << 16) | (fs << 11) | (fd << 6)


@pytest.mark.parametrize("register_pairs, expected", [(5, True), (4, True), (3, False), (0, False)])
def test_register_dominant(register_pairs: int, expected: bool) -> None:
    built = _words(*([_addu(2)] * register_pairs + [_addiu(2, 4, 1)] * (5 - register_pairs)))
    target = _words(*([_addu(3)] * register_pairs + [_addiu(2, 4, 3)] * (5 - register_pairs)))
    facts = symptoms.measure(built, target, ".text")
    assert facts == {"bytes_differ": True, "register_dominant": expected, "sp_offset_only": False}


@pytest.mark.parametrize("rd, rs, rt", [(3, 4, 5), (3, 6, 7), (29, 4, 5)])
def test_register_positions_and_sp_exclusion(rd: int, rs: int, rt: int) -> None:
    facts = symptoms.measure(_words(_addu()), _words(_addu(rd, rs, rt)), ".text")
    assert facts["register_dominant"] is (rd != 29)


@pytest.mark.parametrize("count", [1, 3])
def test_float_register_differences(count: int) -> None:
    facts = symptoms.measure(_words(*([_float(2)] * count)), _words(*([_float(8, 10, 12)] * count)), ".text")
    assert facts == {
        "bytes_differ": True,
        "register_dominant": False,
        "sp_offset_only": False,
        "float_register_differences": count,
    }


@pytest.mark.parametrize("base, expected", [(29, True), (4, False)])
@pytest.mark.parametrize("offset", [-8, 0, 16])
def test_sp_offset_only(base: int, expected: bool, offset: int) -> None:
    facts = symptoms.measure(_words(_load(2, base, offset)), _words(_load(2, base, offset + 4)), ".text")
    assert facts == {"bytes_differ": True, "register_dominant": False, "sp_offset_only": expected}


def test_sp_offset_mixed_with_register_change() -> None:
    facts = symptoms.measure(_words(_load(2, 29, 8)), _words(_load(3, 29, 12)), ".text")
    assert facts == {"bytes_differ": True, "register_dominant": False, "sp_offset_only": False}


@pytest.mark.parametrize("delta", [-20, -16, -12, -8, -4, -2, 0, 2, 4, 8, 12, 16, 20])
@pytest.mark.parametrize("immediate", [-32, 32])
def test_off_four_immediate(delta: int, immediate: int) -> None:
    facts = symptoms.measure(_words(_addiu(2, 4, immediate)), _words(_addiu(2, 4, immediate + delta)), ".text")
    expected = {"bytes_differ": delta != 0}
    if delta:
        expected |= {"register_dominant": False, "sp_offset_only": False}
        if abs(delta) <= 16 and delta % 4 == 0:
            expected["literal_immediates_off_four"] = 1
    assert facts == expected


@pytest.mark.parametrize("built_frame, target_frame", [(16, 32), (32, 16), (16, 16)])
def test_frame_delta(built_frame: int, target_frame: int) -> None:
    # Later prologues must not replace the first detected frame size.
    built = _words(_addiu(29, 29, -built_frame), _addiu(29, 29, -64))
    target = _words(_addiu(29, 29, -target_frame), _addiu(29, 29, -80))
    facts = symptoms.measure(built, target, ".text")
    if built_frame == target_frame:
        assert "frame_delta_bytes" not in facts
    else:
        assert facts["frame_delta_bytes"] == target_frame - built_frame


@pytest.mark.parametrize("other", [_addu(), _addiu(29, 29, 16), _addiu(29, 4, -16)])
def test_frame_delta_requires_two_prologues(other: int) -> None:
    assert "frame_delta_bytes" not in symptoms.measure(_words(_addiu(29, 29, -32)), _words(other), ".text")


@pytest.mark.parametrize("lengths, expected", [((1, 1), 2), ((2, 2), 2), ((1, 2), None), ((0, 1), None)])
def test_repeated_deleted_runs(lengths: tuple[int, int], expected: int | None) -> None:
    # addu, or, xor anchor two separated nop runs without ambiguous alignment.
    anchors = (_addu(), _addu() + 4, _addu() + 5)
    built = _words(anchors[0], *([0] * lengths[0]), anchors[1], *([0] * lengths[1]), anchors[2])
    facts = symptoms.measure(built, _words(*anchors), ".text")
    assert facts.get("repeated_deleted_runs") == expected
    assert "register_dominant" not in facts


def test_replacements_with_different_mnemonics() -> None:
    facts = symptoms.measure(_words(_addu()), _words(_load(2, 29, 8)), ".text")
    assert facts == {"bytes_differ": True, "register_dominant": False, "sp_offset_only": False}


def test_unequal_replacement_lengths_are_not_paired() -> None:
    facts = symptoms.measure(_words(_addu()), _words(_load(2, 29, 8), _load(2, 29, 12)), ".text")
    assert facts == {"bytes_differ": True, "size_delta_bytes": -4}


@pytest.mark.parametrize("section", [".data", ".rodata", ".bss", ".other"])
@pytest.mark.parametrize(
    "built, target, expected",
    [
        (_words(_addu()), _words(_addu(3)), {"bytes_differ": True}),
        (b"same", b"same", {"bytes_differ": False}),
        (b"longer", b"x", {"bytes_differ": True, "size_delta_bytes": 5}),
        (b"", b"four", {"bytes_differ": True, "size_delta_bytes": -4}),
    ],
)
def test_non_text_sections_only_bytes_and_size(monkeypatch, section, built, target, expected) -> None:
    def no_decode(*args, **kwargs):
        pytest.fail("non-text must not be decoded")

    monkeypatch.setattr(symptoms.rabbitizer, "Instruction", no_decode)
    assert symptoms.measure(built, target, section) == expected


@pytest.mark.parametrize("built, target", [(b"x", b"four"), (b"four", b"x"), (b"x", b"y")])
def test_unaligned_text_only_bytes_and_size(monkeypatch, built: bytes, target: bytes) -> None:
    def no_decode(*args, **kwargs):
        pytest.fail("unaligned text must not be decoded")

    monkeypatch.setattr(symptoms.rabbitizer, "Instruction", no_decode)
    expected = {"bytes_differ": built != target}
    if len(built) != len(target):
        expected["size_delta_bytes"] = len(built) - len(target)
    assert symptoms.measure(built, target, ".text") == expected


@pytest.mark.parametrize(
    "built, target, expected",
    [
        (b"", b"", 1.0),
        (b"abcd", b"abcd", 1.0),
        (b"x", b"x", 1.0),
        (b"x", b"", 0.0),
        (b"", b"abcd", 0.0),
        (b"ABCD", b"ABCE", 0.0),
        (b"abX", b"abc", 2 / 3),
        (b"ab", b"abc", 2 / 3),
        (_words(1, 9), _words(1, 2), 0.5),
        (_words(1), _words(1, 2), 0.5),
        (_words(9, 1, 2), _words(1, 2), 1.0 - 2**-53),
        (b"xabc", b"abc", 1.0 - 2**-53),
    ],
)
def test_score_equal_one_and_partial(built: bytes, target: bytes, expected: float) -> None:
    assert symptoms.score(built, target) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("struct S has no member named field", {"missing_struct_member": True}),
        ("invalid use of incomplete type", {"missing_struct_member": True}),
        ("unrelated diagnostic", {}),
        ("size 8 != 12", {"size_delta_bytes": -4}),
        ("size 16 != 12", {"size_delta_bytes": 4}),
        ("bytes differ", {"bytes_differ": True}),
        ("unproved", {"unproved": True}),
        ("no measurement", {"no_measurement": True}),
        ("compile.error", {"compile_failed": True}),
        ("unresolved symbol", {"unresolved": True}),
    ],
)
def test_from_missing_struct_member(text: str, expected: dict) -> None:
    assert symptoms.from_missing([text]) == expected


def test_from_missing_combines_messages() -> None:
    assert symptoms.from_missing([]) == {}
    assert symptoms.from_missing(["bytes differ; size 20 != 16", "compile.error: incomplete type", "unproved"]) == {
        "bytes_differ": True,
        "size_delta_bytes": 4,
        "compile_failed": True,
        "missing_struct_member": True,
        "unproved": True,
    }


def test_merge_rules() -> None:
    first = {
        "bytes_differ": False,
        "compile_failed": True,
        "sp_offset_only": False,
        "size_delta_bytes": 0,
        "frame_delta_bytes": -16,
        "score": 0.8,
        "repeated_deleted_runs": 2,
        "float_register_differences": 1,
        "literal_immediates_off_four": 0,
        "plateau_probes": 2,
        "unknown": True,
    }
    second = {
        "bytes_differ": True,
        "compile_failed": False,
        "sp_offset_only": False,
        "size_delta_bytes": 4,
        "frame_delta_bytes": 32,
        "score": 0.4,
        "repeated_deleted_runs": 3,
        "float_register_differences": 2,
        "literal_immediates_off_four": 0,
        "plateau_probes": 3,
    }
    assert symptoms.merge([first, second, {"score": 0.9}]) == {
        "bytes_differ": True,
        "compile_failed": True,
        "size_delta_bytes": 0,
        "frame_delta_bytes": -16,
        "score": 0.4,
        "repeated_deleted_runs": 5,
        "float_register_differences": 3,
        "plateau_probes": 5,
    }
    assert first["bytes_differ"] is False
    assert symptoms.merge([]) == {}
    assert symptoms.merge([{"plateau_probes": 0, "unproved": False, "alien": 1}]) == {}


@pytest.mark.parametrize("vram", [0, 0x80000400])
def test_diff_has_addresses(vram: int) -> None:
    built, target = _words(_addu(), _addu(2)), _words(_addu(), _addu(3))
    assert symptoms.diff(built, target, vram) == "\n".join(
        [
            "--- target",
            "+++ built",
            "@@ -1,2 +1,2 @@",
            f" {vram:08X}: addu $v0, $a0, $a1",
            f"-{vram + 4:08X}: addu $v1, $a0, $a1",
            f"+{vram + 4:08X}: addu $v0, $a0, $a1",
        ]
    )
    assert symptoms.diff(target, target, vram) == ""
    assert symptoms.diff(b"x", b"y", vram) == ""


def test_diff_one_inserted_instruction_is_one_hunk() -> None:
    nops = [_addu(rd) for rd in range(8, 16)]
    target = _words(0x10400003, *nops)
    built = _words(0x10400004, nops[0], _addu(2), *nops[1:])
    out = symptoms.diff(built, target, 0x80000000).splitlines()
    assert sum(line.startswith("@@") for line in out) == 1
    assert [line for line in out[2:] if line[0] in "+-"] == ["+80000008: addu $v0, $a0, $a1"]
    assert " 80000000: beqz $v0 -> 80000010" in out


def test_shift_reports_the_first_difference_and_the_aligned_counts_past_an_insertion() -> None:
    target = _words(*(_addiu(2, 4, n) for n in range(1, 9)))
    built = _words(*(_addiu(2, 4, n) for n in range(1, 4)), _addiu(9, 9, 99), *(_addiu(2, 4, n) for n in range(4, 8)))
    shift = symptoms.shift(built, target)
    assert shift["first_difference"] == {"index": 3, "built": "addiu $t1, $t1, 0x63", "target": "addiu $v0, $a0, 0x4"}
    assert shift["aligned"] == {"exact": 7, "register_blind": 7} and shift["positional"] == 3  # one word was inserted
    assert symptoms.shift(target, target)["first_difference"] is None


def test_shift_counts_register_blind_alignment_apart_from_the_exact_one() -> None:
    target = _words(_addu(2, 4, 5), _addiu(2, 4, 1), _addu(3, 4, 5))
    built = _words(_addu(8, 4, 5), _addiu(2, 4, 1), _addu(3, 4, 5))  # only the first destination register differs
    assert symptoms.shift(built, target)["aligned"] == {"exact": 2, "register_blind": 3}


def test_register_votes_count_the_target_register_for_each_built_register_in_aligned_instructions() -> None:
    target = _words(_addu(2, 20, 5), _addiu(2, 20, 1))  # the target keeps its value in $s4
    built = _words(_addu(2, 21, 5), _addiu(2, 21, 1))  # the candidate keeps it in $s5
    votes = symptoms.register_votes(built, target)
    assert votes["$s5"] == {"$s4": 2} and votes["$a1"] == {"$a1": 1} and votes["$v0"] == {"$v0": 2}


def _word(op: int, rs: int, rt: int, imm: int) -> bytes:
    return ((op << 26) | (rs << 21) | (rt << 16) | imm).to_bytes(4, "big")


def test_literals_make_a_float_load_of_the_same_value_at_another_address_equal() -> None:
    lui, same, other = _word(0x0F, 0, 1, 0x8000), _word(0x35, 1, 2, 0x3EB0), _word(0x35, 1, 2, 0x2108)
    values = {0x80003EB0: b"A" * 8, 0x80002108: b"A" * 8}
    read = lambda a, n: values.get(a, b"")  # noqa: E731
    assert symptoms.literals(lui + other, lui + same, read, read) == lui + same
    differ = lambda a, n: b"B" * 8 if a == 0x80003EB0 else values.get(a, b"")  # noqa: E731
    assert symptoms.literals(lui + other, lui + same, read, differ) == lui + other
