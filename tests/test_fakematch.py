"""One FAKEMATCH detection for the gate's waivers and the published score: own, inherited and per-version notes."""

from unbake import fakematch

NOTE = "/* FAKEMATCH: a dead clamp keeps the register live */"


def test_a_note_inside_a_function_taints_it_alone():
    text = f"int Own(int a) {{\n    {NOTE}\n    return a;\n}}\nint Caller(void) {{ return Own(1); }}\n"
    assert fakematch.tainted({"src/a.c": text}, "us") == {"Own"}  # a called function is its own code


def test_an_inline_function_passes_its_note_to_every_user_across_files():
    header = f"static inline float depthScale(float d) {{\n    {NOTE}\n    return d;\n}}\n"
    user = (
        '#include "h.h"\nfloat Particle_Build(float d) { return depthScale(d) * 2; }\nint Other(void) { return 0; }\n'
    )
    two = "float Particle_Scale(float d) { return Particle_Build(d); }\n"
    got = fakematch.tainted({"include/h.h": header, "src/a.c": user, "src/b.c": two}, "us")
    assert got == {"depthScale", "Particle_Build"}


def test_a_macro_note_reaches_the_functions_that_expand_it():
    header = f"{NOTE}\n#define CLAMP(x, lo) ((x) < (lo) ? (lo) : (x))\n#define TWICE(x) CLAMP(x, 0) * 2\n"
    user = "int Use(int v) { return TWICE(v); }\nint Clean(int v) { return v; }\n"
    assert fakematch.tainted({"include/m.h": header, "src/u.c": user}, "us") == {"CLAMP", "TWICE", "Use"}


def test_a_note_counts_only_in_the_versions_that_compile_it():
    text = f"int F(void) {{\n#if defined(VERSION_DE)\n    {NOTE}\n#endif\n    return 0;\n}}\n"
    assert fakematch.tainted({"src/f.c": text}, "de") == {"F"}
    assert fakematch.tainted({"src/f.c": text}, "us") == frozenset()
    off = f"int G(void) {{\n#ifdef NON_MATCHING\n    {NOTE}\n#endif\n    return 0;\n}}\n"
    assert fakematch.tainted({"src/g.c": off}, "us") == frozenset()  # the matching build has NON_MATCHING off


def test_the_gate_reads_the_same_note():
    assert fakematch.WAIVER.findall(NOTE) == [" a dead clamp keeps the register live "]
    assert fakematch.WAIVER.findall("/* FAKEMATCH */") == []


SDK_FN = (
    "int Send(int a) {\n    do {\n        if (a) {\n#if defined(__i386__)\n            __builtin_trap();\n#else\n"
    '            __asm__ volatile ("break 1");\n#endif\n        }\n    } while (a);\n    return a;\n}\n'
)


def _sdk(path: str) -> bool:
    return path.startswith("src/sdk/libsn/")


def test_inline_asm_in_a_compiled_function_is_no_c_unless_the_sdk_exception_allows_it():
    """10-25 REV1-ONE-72: Libsn_SendHostCommand is exact through SN's own `break 1`. Inline asm inside a function
    scores like a FAKEMATCH, except in a library folder with only instructions cc1 cannot emit (owner to confirm)."""
    texts = {
        "src/sdk/libsn/a.c": SDK_FN,
        "src/game/b.c": SDK_FN.replace("Send", "GameSend"),
        "src/sdk/libsn/c.c": 'int Add(int a) { __asm__ volatile ("addiu $2, $4, 1"); return a; }\n',
        "src/sdk/libsn/d.c": 'INCLUDE_ASM("seg", Stub);\nint Plain(void) { return 1; }\n',
    }
    got = fakematch.tainted(texts, "us", _sdk)
    assert "Send" not in got and "GameSend" in got and "Add" in got and "Plain" not in got
    assert "Send" in fakematch.tainted(texts, "us")  # no SDK folders known: nothing is allowed
    rows = fakematch.asm_allowed(texts, ["us", "de"], _sdk)
    assert rows == [{"path": "src/sdk/libsn/a.c", "function": "Send", "asm": '"break 1"'}]


def test_the_cheat_gate_reads_one_added_line_the_same_way():
    line = '            __asm__ volatile ("break 1");'
    assert fakematch.sdk_line(line, "src/sdk/libsn/a.c", _sdk) == '"break 1"'
    assert fakematch.sdk_line(line, "src/game/b.c", _sdk) == ""
    assert fakematch.sdk_line('__asm__("addiu $2, $4, 1");', "src/sdk/libsn/a.c", _sdk) == ""
