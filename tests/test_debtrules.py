"""The audit's debt rules (hex-value, raw-data, placeholder-name, stand-in, guard-gap, sdk-name) on slices cut from
RageWars changes the audit must judge: each failed to refuse or wrongly refused before its rule."""

import subprocess

import pytest

from unbake import config, debtrules, human, policy
from unbake.contracts import Finding

SETTINGS = config.audit_settings(None)

# RageWars 9faab54a8d, src/AI/PLAYER/PLAYER_DEATH.CPP: a cleanup rewrote 0x11 to 17 beside a held 0x140, which the table
# no longer reads as one hex-only argument list. The 0x140 was already in the base: the change adds no hex.
HELD_BEFORE = "void f(void) {\n    CScene_StartEventClips(&g_world, FIELD(void *, p, 0x698) + 0x140, 0x11, zero);\n}\n"
HELD_AFTER = "void f(void) {\n    CScene_StartEventClips(&g_world, FIELD(void *, p, 1688) + 0x140, 17, zero);\n}\n"
# RageWars be7ae9a32d, include/MULTI/PLAYER_SLOTS.H: a new header spelled array sizes in hex.
NEW_SIZES = "typedef struct PlayerSlots {\n    u8 names[0x58];\n    u8 rest[0xB68];\n} PlayerSlots;\n"
# RageWars 4654aea3c3: assert code kept as raw words of MIPS code with its `jr ra`.
CODE_WORDS = (
    "u32 g_AssertSlotIndex[] = {0x27BDFFE8, 0xAFBF0014, 0x8C820000, 0x24420001, 0xAC820000, 0x8FBF0014, 0x27BD0018,"
    " 0x03E00008, 0};\n"
)


def _repo(tmp_path, monkeypatch, files):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        subprocess.run([*git, *args], check=True)


def _refused(tmp_path, changes):
    for path, text in changes.items():
        (tmp_path / path).write_text(text)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    with pytest.raises(policy.Refusal) as refused:
        policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    return [(f.key, f.severity) for f in refused.value.findings]


def _lines(code, check, *args):
    return getattr(debtrules, check)(policy.scopes(code)["code"], SETTINGS, *args)


def test_a_hex_value_in_call_arguments_and_arithmetic_is_dec_and_a_mask_is_kept():
    code = "void f(void) {\n    g(a + 0x2BC);\n    x = y & 0xFF;\n    z = w + 0x11;\n}\n"
    assert _lines(code, "hex_values", "DEC") == {2: "0x2BC (id)", 4: "0x11 (value)"}


def test_hex_review_classes_are_not_dec():
    assert _lines("int x = 0x400;\n", "hex_values", "DEC") == {}
    assert _lines("void f(void) {\n    x = 0x400;\n}\n", "hex_values", "REVIEW") == {2: "0x400 (pow2)"}


def test_the_audit_refuses_a_new_header_that_spells_array_sizes_in_hex(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, {"include/a.h": "int x;\n"})
    assert ("source.hex-value", "Refuse") in _refused(tmp_path, {"include/p.h": NEW_SIZES})


def test_the_audit_does_not_blame_a_hex_literal_the_base_already_held(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, {"src/a.c": HELD_BEFORE})
    (tmp_path / "src/a.c").write_text(HELD_AFTER)
    assert policy.audit_command({"project": str(tmp_path), "against": "HEAD"})["warnings"] == {}


def test_a_hex_literal_with_a_suffix_is_held_like_any_other(tmp_path, monkeypatch):
    # RageWars aab9c527cc: a suffixed literal (0x10u) has no word boundary after its digits; the audit crashed on it.
    _repo(tmp_path, monkeypatch, {"src/a.c": "void f(void) {\n    g(0x2BCu, 0x11);\n}\n"})
    (tmp_path / "src/a.c").write_text("void f(void) {\n    g(0x2BCu, 17);\n}\n")
    assert policy.audit_command({"project": str(tmp_path), "against": "HEAD"})["warnings"] == {}


def test_raw_words_that_are_mips_code_are_certain_and_a_pad_class_is_allowed():
    assert _lines(CODE_WORDS, "raw_data", True) == {1: "g_AssertSlotIndex is MIPS code"}
    assert _lines("u8 g_TrailingBytes[] = {1, 2};\nu8 g_pad[] = {0, 0, 0, 0};\n", "raw_data", True) == {}


def test_raw_words_that_look_like_floats_are_a_judgment_not_a_refusal():
    code = "u32 g_t[] = {0x3F800000, 0x40000000, 0x40400000};\n"
    assert _lines(code, "raw_data", True) == {}
    assert _lines(code, "raw_data", False) == {1: "g_t is a float table"}


def test_a_long_nul_terminated_byte_array_is_a_string():
    text = "char g_s[] = {" + ", ".join(str(b) for b in b"Hello, world, here\0") + "};\n"
    assert _lines(text, "raw_data", True) == {1: "g_s is a string"}


def test_placeholder_names_are_found_by_address_or_numbered_suffix():
    code = "void func_80012345(void);\nextern u8 D_80031234;\nint g_Block17;\nint unk_4;\nint ok;\n"
    assert _lines(code, "placeholder_names") == {1: "func_80012345", 2: "D_80031234", 3: "g_Block17", 4: "unk_4"}


def test_a_placeholder_folder_is_named_by_its_first_component():
    assert debtrules.placeholder_path("src/func_80012345.c", SETTINGS) == "func_80012345.c"
    assert debtrules.placeholder_path("src/ok/a.c", SETTINGS) is None


def test_the_audit_refuses_a_stand_in_a_non_matching_note_and_a_placeholder_path(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, {"src/a.c": "int x;\n"})
    stand_in = 'INCLUDE_ASM("asm/a", f);\n#ifdef NON_MATCHING\nint y;\n#endif\n'
    keys = {key for key, _ in _refused(tmp_path, {"src/func_80012345.c": stand_in})}
    assert {"source.stand-in", "source.non-matching", "source.placeholder-name"} <= keys


def test_sdk_names_belong_to_library_roots_and_reserved_names_stay_there():
    inside = "void myHelper(void) {\n}\nvoid osSetTimer(void) {\n}\n"
    assert _lines(inside, "sdk_names", True) == {1: "myHelper is no SDK name"}
    assert _lines("void __osFoo(void) {\n}\nvoid Plain(void) {\n}\n", "sdk_names", False) == {
        1: "__osFoo is an SDK-reserved name"
    }


def test_every_finding_has_one_shape_and_the_tally_counts_codes_by_severity():
    refuse = Finding("source.stand-in", "INCLUDE_ASM stands in", path="src/a.c", line=3, blocking=True)
    warn = Finding("source.hex-review", "0x400 (pow2)", path="src/a.c", line=9, blocking=False)
    assert set(human.standard(refuse)) >= {"Code", "Severity", "File", "Line", "Message", "NextStep"}
    assert human.standard(warn)["Severity"] == "Warn"
    tally = human.tally([refuse, warn, warn])
    assert (tally["Refuse"], tally["Warn"]) == (1, 2)
    assert tally["Codes"]["source.hex-review"] == {"Warn": 2}
    assert human.tally_line(tally) == "summary: 1 Refuse, 2 Warn (source.stand-in 1 Refuse, source.hex-review 2 Warn)"


def test_a_rule_refuses_only_where_a_match_is_certain():
    rules = {r["id"]: r["severity"] for r in config.load_resource("rules.toml")["rule"]}
    exact = ("hex-value", "raw-data", "placeholder-name", "stand-in", "non-matching", "fakematch-note", "sdk-name")
    fuzzy = ("hex-review", "raw-data-review", "raw-offset-cast", "guard-gap")
    assert {rules[r] for r in exact} == {"Refuse"}
    assert {rules[r] for r in fuzzy} == {"Warn"}


_BLOCK = "g_LinktablesEncodedWordsBlock15"


def test_a_size_assertion_over_an_existing_name_is_no_new_placeholder_name():
    held = f"Gfx {_BLOCK}[3] = {{ 0 }};\n"
    after = held + f"typedef char {_BLOCK}SizeCheck[(sizeof({_BLOCK}) == 12) ? 1 : -1];\nu32 n = sizeof({_BLOCK});\n"
    assert _lines(held, "placeholder_names") == {1: _BLOCK}
    assert _lines(after, "placeholder_names") == {1: _BLOCK}


def test_version_copies_of_one_record_are_one_identity():
    code = (
        "#if VERSION_US\ntypedef struct {\n    u32 unk_4;\n} g_Block17;\n"
        "#else\ntypedef struct {\n    u32 unk_4;\n} g_Block17;\n#endif\n"
    )
    assert _lines(code, "placeholder_names") == {3: "unk_4", 4: "g_Block17"}
    copies = f"#if VERSION_US\nextern u32 {_BLOCK};\n#else\nextern u32 {_BLOCK};\n#endif\n"
    assert _lines(copies, "placeholder_names") == {2: _BLOCK}


def test_distinct_records_keep_distinct_placeholder_members():
    code = "typedef struct {\n    u32 unk_4;\n} A;\ntypedef struct {\n    u32 unk_4;\n} B;\n"
    code += "struct C {\n    u32 unk_4;\n};\n"
    assert set(_lines(code, "placeholder_names")) == {2, 5, 8}


def test_a_new_placeholder_definition_and_a_reference_to_it_count_once_at_the_definition():
    code = "void f(void) {\n    D_80031234 = 1;\n    g(D_80031234);\n}\nint D_80031234;\nint func_80012345(void);\n"
    assert _lines(code, "placeholder_names") == {5: "D_80031234", 6: "func_80012345"}
