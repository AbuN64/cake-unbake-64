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
