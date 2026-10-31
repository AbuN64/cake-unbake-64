"""types: accesses read from synthetic assembly, inferred layouts, the header diff and request settling."""

from pathlib import Path

import pytest

from unbake import accesses, cheaders, layouts
from unbake.contracts import Refusal


def asm(*bodies: tuple[str, str]) -> str:
    """Function text: each body line is an instruction (with its operands), one per line."""
    out = []
    pc = 0x80200000
    for name, body in bodies:
        out.append(f"glabel {name}")
        for line in body.strip().splitlines():
            line = line.strip()
            if line.endswith(":"):
                out.append(f"  {line}")
                continue
            out.append(f"    /* 0 {pc:08X} 00000000 */  {line}")
            pc += 4
        out.append(f"endlabel {name}")
    return "\n".join(out)


def found(text: str) -> list[accesses.Access]:
    return [a for f in accesses.functions(text) for a in f.accesses]


def brief(items: list[accesses.Access]) -> list[tuple]:
    return [(a.base[0], a.offset, a.width, a.kind, a.store) for a in items]


def test_offsets_widths_and_signedness_of_an_argument():
    text = asm(
        (
            "f",
            """
            lb $v0, 0x0($a0)
            lbu $v0, 0x1($a0)
            lh $v0, 0x2($a0)
            lhu $v0, 0x4($a0)
            sw $v0, 0x8($a0)
            lwc1 $f0, 0xC($a0)
            swc1 $f0, 0x10($a0)
            sb $v0, 0x14($a0)
            ldc1 $f0, 0x18($a0)
            jr $ra
             nop
            """,
        )
    )
    assert brief(found(text)) == [
        ("arg", 0, 1, "s8", False),
        ("arg", 1, 1, "u8", False),
        ("arg", 2, 2, "s16", False),
        ("arg", 4, 2, "u16", False),
        ("arg", 8, 4, "i32", True),
        ("arg", 0xC, 4, "f32", False),
        ("arg", 0x10, 4, "f32", True),
        ("arg", 0x14, 1, "i8", True),
        ("arg", 0x18, 8, "f64", False),
    ]


def test_global_symbol_hi_lo_and_addend_and_address_taken():
    text = asm(
        (
            "f",
            """
            lui $v0, %hi(D_800C0000)
            lw $v1, %lo(D_800C0000)($v0)
            lui $a1, %hi(D_800C0010 + 0x4)
            sh $zero, %lo(D_800C0010 + 0x4)($a1)
            lui $a2, %hi(D_800C0100)
            addiu $a2, $a2, %lo(D_800C0100)
            lbu $v0, 0x20($a2)
            jr $ra
             nop
            """,
        )
    )
    assert [(a.base, a.offset, a.width) for a in found(text)] == [
        (("sym", "D_800C0000"), 0, 4),
        (("sym", "D_800C0010"), 4, 2),
        (("sym", "D_800C0100"), 0x20, 1),
    ]


def test_pointer_chain_and_pointer_proof_from_dereference():
    text = asm(
        (
            "f",
            """
            lw $v0, 0x10($a0)
            lw $v1, 0x4($v0)
            lw $a1, 0x20($a0)
            jr $ra
             nop
            """,
        )
    )
    function = accesses.functions(text)[0]
    assert [(a.base, a.offset) for a in function.accesses] == [
        (("arg", "f", 0), 0x10),
        (("load", ("arg", "f", 0), 0x10), 4),
        (("arg", "f", 0), 0x20),
    ]
    assert [(p.base, p.offset, p.why) for p in function.pointers] == [(("arg", "f", 0), 0x10, "deref")]


def test_moves_and_adds_are_followed_and_a_call_clobbers_temporaries():
    text = asm(
        (
            "f",
            """
            addu $s0, $a0, $zero
            addiu $t0, $s0, 0x40
            lw $v0, 0x8($t0)
            jal g
             addu $a1, $zero, $zero
            lw $v0, 0x8($s0)
            lw $v1, 0x8($t0)
            jr $ra
             nop
            """,
        )
    )
    assert [(a.base[0], a.offset) for a in found(text)] == [("arg", 0x48), ("arg", 8)]


def test_a_register_with_two_values_at_a_join_is_dropped_and_equal_ones_kept():
    text = asm(
        (
            "f",
            """
            beqz $a2, .Lskip
             addu $s0, $a0, $zero
            addu $s0, $a1, $zero
            .Lskip:
            lw $v0, 0x4($s0)
            lw $v1, 0x8($a0)
            jr $ra
             nop
            """,
        )
    )
    assert [(a.base, a.offset) for a in found(text)] == [(("arg", "f", 0), 8)]


def test_indexed_element_is_its_own_base_and_stack_is_ignored():
    text = asm(
        (
            "f",
            """
            sll $v0, $a1, 4
            addu $v0, $a0, $v0
            lw $v1, 0x4($v0)
            sw $v1, 0x10($sp)
            jr $ra
             nop
            """,
        )
    )
    assert [(a.base, a.offset) for a in found(text)] == [(("idx", ("arg", "f", 0), 0), 4)]


def test_a_pointer_stepping_through_an_array_becomes_an_indexed_base():
    text = asm(
        (
            "f",
            """
            addu $v0, $a0, $zero
            .Lloop:
            lw $v1, 0x4($v0)
            bnez $v1, .Lloop
             addiu $v0, $v0, 0x20
            jr $ra
             nop
            """,
        )
    )
    assert [(a.base, a.offset) for a in found(text)] == [(("idx", ("arg", "f", 0), 0), 4)]


def test_storing_a_symbol_address_proves_a_pointer_field():
    text = asm(
        (
            "f",
            """
            lui $v0, %hi(D_800C0000)
            addiu $v0, $v0, %lo(D_800C0000)
            sw $v0, 0x1C($a0)
            jr $ra
             nop
            """,
        )
    )
    assert [(p.offset, p.why) for p in accesses.functions(text)[0].pointers] == [(0x1C, "stored")]


HEADER = """
typedef struct Thing Thing;
struct Thing {
    s32 a;
    s16 b;
    u8 c;
    u8 flag;
    char pad8[0x10 - 0x8];
    s32 count;
    f32 scale;
    u32 *list;
    s32 tail;
};
extern void use(Thing *arg0, int x);
extern void f(Thing *arg0);
extern Thing D_800C0000;
"""

USER = """
lw $v0, 0x0($a0)
lhu $v0, 0x4($a0)
lb $v0, 0x7($a0)
sw $v0, 0x8($a0)
lw $v0, 0x10($a0)
lw $v1, 0x18($a0)
lw $a2, 0x0($v1)
lwc1 $f0, 0x1C($a0)
lw $v0, 0x24($a0)
jal use
 addu $a0, $a0, $zero
jr $ra
 nop
"""


def project(tmp_path: Path, body: str) -> Path:
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "T.H").write_text(HEADER)
    (tmp_path / "asm" / "v1").mkdir(parents=True)
    (tmp_path / "asm" / "v1" / "f.s").write_text(body)
    (tmp_path / "versions" / "v1").mkdir(parents=True)
    (tmp_path / "versions" / "v1" / "symbol_addrs.txt").write_text(
        "D_800C0000 = 0x800C0000; //\nD_800C0100 = 0x800C0100; //\n"
    )
    return tmp_path


def headers_of(root: Path) -> cheaders.Headers:
    headers = cheaders.Headers()
    headers.read(root)
    return headers


def test_header_struct_size_offsets_and_padding_arrays():
    headers = cheaders.Headers()
    headers.add(HEADER)
    thing = headers.struct("Thing")
    assert thing is not None and thing.complete and thing.size == 0x20
    assert [(f.name, f.offset) for f in thing.fields] == [
        ("a", 0), ("b", 4), ("c", 6), ("flag", 7), ("pad8", 8),
        ("count", 0x10), ("scale", 0x14), ("list", 0x18), ("tail", 0x1C),
    ]  # fmt: skip
    assert headers.prototypes["use"].params == ["Thing * arg0", "int x"]


def test_header_reader_handles_anonymous_unions_nested_records_and_sizeof():
    headers = cheaders.Headers()
    headers.add(
        """
        typedef struct Inner { s16 x; s16 y; } Inner;
        typedef struct Outer {
            u8 head;
            union { struct { s32 p; s32 q; }; f32 f[2]; };
            Inner pts[3];
            char pad[0x40 - 0x1C - sizeof(s32)];
        } Outer;
        """
    )
    outer = headers.struct("Outer")
    assert outer is not None and outer.complete
    assert [(f.name, f.offset) for f in outer.fields][:5] == [("head", 0), ("p", 4), ("q", 8), ("f", 4), ("pts", 12)]
    assert headers.locate(outer, 14) is not None and headers.locate(outer, 14)[0] == "pts[0].y"


def test_inferred_layout_pointer_gap_and_header_disagreements(tmp_path: Path):
    root = project(tmp_path, asm(("f", USER), ("use", "jr $ra\n nop")))
    inferred = layouts.infer(root, "v1")
    record = {r["name"]: r for r in inferred["records"]}["Thing"]
    assert record["kind"] == "struct"
    assert [(f["offset"], f["width"]) for f in record["fields"]] == [
        (0, 4), (4, 2), (7, 1), (8, 4), (0x10, 4), (0x18, 4), (0x1C, 4), (0x24, 4),
    ]  # fmt: skip
    pointer = next(f for f in record["fields"] if f["offset"] == 0x18)
    assert pointer["pointer"] and pointer["type"] == "void *" and pointer["name"] == "list"
    assert record["gaps"] == [
        {"offset": 6, "size": 1},
        {"offset": 0xC, "size": 4},
        {"offset": 0x14, "size": 4},
        {"offset": 0x20, "size": 4},
    ]
    issues = {(i["offset"], i["kind"]) for i in record["diff"]}
    assert (7, "signedness") in issues  # lb against u8 flag
    assert (8, "hidden-in-padding") in issues  # sw into pad8
    assert (0x1C, "float-as-int") in issues
    assert (0x24, "past-end") in issues
    assert not any(i["offset"] == 0x18 for i in record["diff"])  # list is already a pointer


def test_requests_are_settled_from_evidence_and_the_rest_say_what_is_missing(tmp_path: Path):
    root = project(tmp_path, asm(("f", USER), ("use", "jr $ra\n nop")))
    text = "\n".join(
        [
            "- SETTLED main abc (done): X: `Thing` +0x10 s32",
            "- A: `Thing` +0x10/+0x18 s32 count and pointer; f loads both",
            "- A: `Thing` +0x10 and +0x14 f32",
            "- A: `Thing` +0x10 likely a role (semantic identity unresolved)",
            "- A: `Nope` +0x10",
            "- A: `Thing` has no offset at all",
        ]
    )
    result = layouts.settle(layouts.parse_requests(text), layouts.infer(root, "v1"), headers_of(root))
    assert [r["status"] for r in result["rows"]] == ["answered", "partial", "layout", "no-record", "no-offset"]
    assert result["settled_skipped"] == 1
    assert result["rows"][0]["evidence"][1]["pointer"] is True
    assert result["rows"][1]["missing"] == ["no access at +0x14"]


def test_a_cited_function_answers_a_request_whose_type_has_no_accesses(tmp_path: Path):
    root = project(tmp_path, asm(("func_80200000_de", "lbu $v0, 0x12($a1)\njr $ra\n nop")))
    text = "- A: func_80200000_de / Mystery +0x12 u8 flag"
    result = layouts.settle(layouts.parse_requests(text), layouts.infer(root, "v1"), headers_of(root))
    row = result["rows"][0]
    assert (row["status"], row["via"]) == ("answered-via-function", "function-argument")
    assert row["evidence"][0]["type"] == "u8"


def test_types_command_refuses_a_project_without_assembly(tmp_path: Path):
    with pytest.raises(Refusal):
        layouts.types_command({"project": tmp_path})


FUNCTIONS = {
    "nonmatchings/main/func_80200000.s": ("func_80200000", "lw $v0, 0x8($a0)\njr $ra\n nop"),
    "nonmatchings/lib/func_80200100.s": ("func_80200100", "lw $v0, 0x4($a1)\njr $ra\n nop"),
}


def nested_tree(root: Path) -> Path:
    for rel, body in FUNCTIONS.items():
        file = root / "asm/us-rev1" / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(asm(body))
    return root


def test_collect_finds_functions_nested_under_nonmatchings(tmp_path: Path):
    root = nested_tree(tmp_path)
    assert sorted(f.name for f in layouts.collect(root, "us-rev1")) == ["func_80200000", "func_80200100"]


def test_the_function_list_reads_lines_or_a_symbol_column(tmp_path: Path):
    plain = tmp_path / "a.txt"
    plain.write_text("# boot\nfunc_a\n\nfunc_b\nfunc_a\n")
    assert layouts.read_functions(plain) == ["func_a", "func_b"]
    table = tmp_path / "b.tsv"
    table.write_text("order\tsymbol\tclass\n1\tfunc_c\tmatched\n2\tfunc_d\topen\n")
    assert layouts.read_functions(table) == ["func_c", "func_d"]
    table.write_text("order\tname\n1\tx\n")
    with pytest.raises(Refusal):
        layouts.read_functions(table)


def test_functions_limit_the_records_to_those_a_listed_function_touches(tmp_path: Path):
    root = nested_tree(tmp_path)
    (root / "boot.txt").write_text("func_80200000\nmissing\n")
    out = layouts.types_command({"project": root, "all": True, "functions": root / "boot.txt"})
    assert out["listed_functions"] == 2
    assert out["records"] and all(r["listed_functions"] == ["func_80200000"] for r in out["records"])
    everything = layouts.types_command({"project": root, "all": True})
    assert len(everything["records"]) > len(out["records"])


def test_order_of_a_module_linked_in_two_orders_by_two_link_layouts_names_the_moved_unit() -> None:
    from unbake import mergecheck

    names = [f"u{i}" for i in range(6)]
    moved_in_us_rev1 = [names[1], names[0], *names[2:]]

    def seq(order: list[str]) -> list[dict]:
        return [{"o": ".game_data", "k": "entry", "u": u, "cls": "rodata", "n": 4} for u in order]

    found = mergecheck.verdicts(
        [(u, "M") for u in names], {"de": seq(names), "us-rev1": seq(moved_in_us_rev1)}, {}, lambda unit: "cc"
    )
    assert found[0]["hard"] == []
    assert [r for r in found[0]["soft"] if r.startswith("order")] == [
        "order: us-rev1 rodata links u1 elsewhere than the other versions"
    ]


def test_sections_parts_real_rows_pakinterfacescreen_data_of_one_version_in_the_extra_game_data_section() -> None:
    from unbake import mergecheck

    names = [f"d{i}" for i in range(11)]

    def seq(extra: str) -> list[dict]:
        return [
            {"o": extra if u == "d10" else ".game_data", "k": "entry", "u": u, "cls": "data", "n": 4} for u in names
        ]

    found = mergecheck.verdicts(
        [(u, "M") for u in names], {"us": seq(".game_data"), "us-rev1": seq(".game_data_E1720")}, {}, lambda unit: "cc"
    )[0]
    assert found["verdict"] == "separate"
    assert [p["units"] for p in found["parts"]["data"]] == [names[:10], ["d10"]]
    assert found["parts"]["data"][1]["sections"] == {"us": ".game_data", "us-rev1": ".game_data_E1720"}
    assert any(h.startswith("sections: data parts: 2: .game_data [d0, d1") for h in found["hard"])
