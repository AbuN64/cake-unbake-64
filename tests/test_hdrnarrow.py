"""An edit of a header reaches only the includers whose compile key can change (no false skips)."""

import os
import subprocess
import time
from types import SimpleNamespace

from unbake import hdrnarrow as hn

GUARD = '#ifndef A_H\n#define A_H\n#include "TYPES.H"\n{}\n#endif\n'


def test_items_and_defs_follow_hdrkeys_drop_rule():
    p = hn.parse(
        GUARD.format(
            "struct Foo;\ntypedef struct Foo Foo;\nstruct Foo { u8 a; void (*f)(int); };\n"
            "extern u8 D_1[0x10];\nextern f32 Math_Sqrtf(f32 x);\ntypedef enum { E_A, E_B } Kind;\n"
        )
    )
    assert not p.why
    got = [hn.defs(i) for i in p.items]
    assert got == [
        None,
        None,
        None,
        {"Foo"},
        {"Foo"},
        {"Foo"},
        {"D_1"},
        {"Math_Sqrtf"},
        {"Kind", "E_A", "E_B"},
        None,
    ]  # the guard's #ifndef/#define, the include and the #endif are fixed items


def test_anything_hdrkey_keeps_always_or_this_parser_does_not_know_is_not_droppable():
    def last(text):
        p = hn.parse(GUARD.format(text))
        assert not p.why, text
        return hn.defs([i for i in p.items if i[0] != "__PP__"][-1])  # the last item before #endif

    for text in ("u8 D_2;", "struct S { u8 a; } s;", "static inline int f(void) { return 1; }", "struct { int a; };"):
        assert last(text) is None, text
    assert last("extern int a, b;") == {"a", "b"}  # a comma list: every name, a safe superset
    assert last('extern "C" { }') is None  # never droppable, whatever item it runs into
    assert hn.parse("#ifdef X\nint a;\n#endif\n").items == [
        ("__PP__", '"ifdef X"'),
        ("int", "a"),
        ("__PP__", '"endif"'),
    ]
    assert hn.defs(hn.parse("#define M 1\n").items[0]) is None  # a directive never drops
    assert hn.parse(GUARD.format("int f(void) { return 1; }\nextern u8 D_3;\n")).items[-2] == ("extern", "u8", "D_3")


def test_changed_defs_names_removed_and_added_items_only():
    old = GUARD.format("extern u8 D_1[8];\nextern u8 D_2;\ntypedef struct W { u8 a; } W;\n")
    new = GUARD.format("extern f32 D_1;\nextern u8 D_2;\ntypedef struct W { u8 a; u8 b; } W;\n")
    assert hn.changed_defs(old, new) == ({"D_1", "W"}, "")
    assert hn.changed_defs(old, old.replace("D_2;", "D_2; /* note */")) == (set(), "")  # a comment changes nothing
    assert hn.changed_defs(None, new)[0] is None  # a new header includes TYPES.H: no pristine unit includes it anyway
    got, why = hn.changed_defs(old, old.replace('"TYPES.H"', '"OTHER.H"'))
    assert got is None and "droppable" in why  # an include changed
    got, why = hn.changed_defs(old, old + "int g(void) { return 0; }\n")
    assert got is None
    got, why = hn.changed_defs(old, old.replace("#endif", "#if X\n#endif\n#endif"))
    assert got is None and "__PP__" in why  # a directive added or changed: not analysable


def test_an_edit_inside_a_version_conditional_names_only_what_it_changes():
    """A version-conditional header (61 `#if` lines in include/COMMON) was refused whole, so a one-struct edit of a
    header 4200 units read rebuilt all of them in the stamp check (385 CPU s, 10-25)."""
    body = "#if defined(VERSION_EU)\nstruct W { u8 a; };\n#else\nstruct W { u16 a; };\n#endif\nextern u8 D_1;\n"
    old = GUARD.format(body)
    assert hn.changed_defs(old, old.replace("u8 a;", "s8 a;")) == ({"W"}, "")
    assert hn.changed_defs(old, old.replace("extern u8 D_1;", "extern s8 D_1;")) == ({"D_1"}, "")
    got, why = hn.changed_defs(old, old.replace("VERSION_EU", "VERSION_DE"))
    assert got is None and "__PP__" in why
    got, why = hn.changed_defs(old, old.replace("#else", "#define W_OTHER 1\n#else"))
    assert got is None


def git(d, *a):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=d, capture_output=True, text=True, check=True
    ).stdout


def tree(tmp_path, units):
    """A minted-looking tree: include/H.H at mint, pristine units with a .i and a key (UNITS: name -> key text)."""
    (tmp_path / "include").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "include/H.H").write_text(GUARD.format("extern u8 D_1[8];\nextern u8 D_2;\n"))
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "m")
    head = git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "build/v/src").mkdir(parents=True)
    for name, key in units.items():
        i = tmp_path / f"build/v/src/{name}.i"
        i.write_text("full preprocessed text")
        if key is not None:
            (tmp_path / f"build/v/src/{name}.i.key").write_text(key)
        (tmp_path / f"build/v/src/{name}.i.ok").write_text("")
    marker = tmp_path / "build/.pool-extracted-at"
    marker.write_text(head + "\n")
    t = time.time() + 5  # the marker is newer than every unit file
    os.utime(marker, (t, t))
    return head


FMT = "build/{v}/src/{unit}.o"


def narrow(tmp_path, units, headers=("include/H.H",)):
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))

    def now(h):
        try:
            return (tmp_path / h).read_text()
        except OSError:
            return None

    return hn.narrow(config, "v", list(headers), units, FMT, now)


def test_narrow_skips_a_unit_whose_key_names_nothing_changed_and_keeps_every_other(tmp_path):
    tree(tmp_path, {"a.CPP": "int f(void){return D_2;}", "b.CPP": "int g(void){return D_1[0];}", "nokey.CPP": None})
    (tmp_path / "include/H.H").write_text(GUARD.format("extern f32 D_1;\nextern u8 D_2;\n"))
    keep, stats = narrow(tmp_path, ["a.CPP", "b.CPP", "nokey.CPP", "gone.CPP"])
    assert keep == ["b.CPP", "nokey.CPP", "gone.CPP"] and stats["skipped"] == 1  # no key: kept, as an absent unit is


def test_narrow_keeps_everything_on_doubt(tmp_path):
    tree(tmp_path, {"a.CPP": "int f(void){return 1;}"})
    units = ["a.CPP"]
    (tmp_path / "include/H.H").write_text(GUARD.format("extern f32 D_1;\nextern u8 D_2;\n"))
    assert narrow(tmp_path, units)[0] == []
    # a macro of that name anywhere: could rename the declaration inside the unit
    (tmp_path / "src/m.h").write_text("#define D_1 D_1_de\n")
    git(tmp_path, "add", "src/m.h")
    keep, stats = narrow(tmp_path, units)
    assert keep == units and "macro" in stats["why"]
    git(tmp_path, "rm", "-qf", "src/m.h")
    # a unit rebuilt after the mint is not pristine
    i_ok = tmp_path / "build/v/src/a.CPP.i.ok"
    later = time.time() + 60
    os.utime(i_ok, (later, later))
    assert narrow(tmp_path, units)[0] == units
    # the tree moved on from the commit the keys were made at
    git(tmp_path, "commit", "-qm", "moved", "--allow-empty")
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/n.h").write_text("extern int N;\n")
    git(tmp_path, "add", "src/n.h")
    git(tmp_path, "commit", "-qm", "more")
    keep, stats = narrow(tmp_path, units)
    assert keep == units and "not the one" in stats["why"]
    # no marker: not a pool tree
    (tmp_path / "build/.pool-extracted-at").unlink()
    assert narrow(tmp_path, units)[0] == units


def test_narrow_keeps_a_unit_whose_key_names_a_removed_or_added_definition(tmp_path):
    tree(tmp_path, {"u.CPP": "extern u8 D_2;\nint f(void){return D_2;}", "w.CPP": "int w(void){return 0;}"})
    (tmp_path / "include/H.H").write_text(GUARD.format("extern u8 D_1[8];\nextern f32 D_2;\n"))
    keep, _ = narrow(tmp_path, ["u.CPP", "w.CPP"])
    assert keep == ["u.CPP"]


def test_the_build_marker_makes_units_built_at_the_mint_pristine(tmp_path):
    head = tree(tmp_path, {"a.CPP": "int f(void){return D_2;}", "b.CPP": "int g(void){return D_1[0];}"})
    # the pool builds after it extracts: the units' .i.ok are newer than the extraction marker
    later = time.time() + 30
    for name in ("a.CPP", "b.CPP"):
        os.utime(tmp_path / f"build/v/src/{name}.i.ok", (later, later))
    (tmp_path / "include/H.H").write_text(GUARD.format("extern f32 D_1;\nextern u8 D_2;\n"))
    assert narrow(tmp_path, ["a.CPP", "b.CPP"])[0] == ["a.CPP", "b.CPP"]
    built = tmp_path / "build/.pool-built-at"
    built.write_text(head + "\n")
    os.utime(built, (later + 5, later + 5))
    assert narrow(tmp_path, ["a.CPP", "b.CPP"])[0] == ["b.CPP"]


def test_a_stamp_rebuilt_after_the_mint_from_the_same_bytes_is_pristine_by_its_record(tmp_path):
    """A reused stamp tree: units its last stamp rebuilt carry a later date than the pool's marker, but their record
    (date + key of the files the .d names) says they were made from the base's bytes, so they still narrow."""
    import hashlib
    import json

    tree(tmp_path, {"a.CPP": "int f(void){return D_2;}", "b.CPP": "int g(void){return D_1[0];}"})
    old_h = (tmp_path / "include/H.H").read_bytes()
    (tmp_path / "src/a.CPP").write_text("a\n")
    later = time.time() + 60
    lines = []
    for name, header_bytes in (("a.CPP", old_h), ("b.CPP", b"other bytes")):
        stamp = tmp_path / f"build/v/src/{name}.i.ok"
        os.utime(stamp, (later, later))  # rebuilt after the marker: not pristine by date
        (tmp_path / f"build/v/src/{name}.d").write_text(f"build/v/src/{name}.i.ok: src/a.CPP include/H.H\n")
        key = hashlib.sha1(" ".join(hashlib.sha1(b).hexdigest() for b in (b"a\n", header_bytes)).encode()).hexdigest()
        lines.append(json.dumps([f"build/v/src/{name}.i.ok", stamp.stat().st_mtime_ns, key]) + "\n")
    (tmp_path / "build/.inputs.log").write_text("".join(lines))
    (tmp_path / "include/H.H").write_text(GUARD.format("extern f32 D_1;\nextern u8 D_2;\n"))
    keep, stats = narrow(tmp_path, ["a.CPP", "b.CPP"])
    assert keep == ["b.CPP"] and stats["skipped"] == 1  # a: made from the base bytes; b: its record holds other bytes
    os.utime(tmp_path / "build/v/src/a.CPP.i.ok", (later + 1, later + 1))  # touched since its record: not trusted
    assert narrow(tmp_path, ["a.CPP", "b.CPP"])[0] == ["a.CPP", "b.CPP"]
