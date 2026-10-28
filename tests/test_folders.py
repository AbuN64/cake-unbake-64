"""folders: rules, the plan's map, the config rewrite, carried patches, and plan/apply/carry/verify on a project."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import fixture
import pytest

from unbake import folders
from unbake.contracts import Config, Refusal

RULES = """\
schema = 1
libraries = []

[[rule]]
match = "src/"
to = "src/game/"

[[rule]]
match = "src/IO/"
to = "src/sdk/io/"
row = { kind = "dir", toolchain = "gcc-sn" }

[[rule]]
match = "src/IO/c.CPP"
to = "src/game/IO/"

[[rule]]
match = "src/libgcc/"
to = "src/sdk/libgcc/"

[[row]]
path = "src/sdk/rnc/"
kind = "library"
origin = "rnc"
"""
TRACKED = [
    "src/IO/a.CPP",
    "src/IO/b.CPP",
    "src/IO/c.CPP",
    "src/main.c",
    "src/libgcc/g.c",
    "src/libgcc/g.h",
    "README.md",
]
CONFIG = """\
schema = 4
name = "T"
toolchain = "gcc-def"
libraries = ["src/libgcc/"]
cppflags = ["-DX", "-Isrc/libgcc"]

[file."src/IO/a.CPP"]
toolchain = "gcc-sn"
cflags = []

[file."src/libgcc/g.c"]
cflags = ["-mfp32", "-Isrc/libgcc"]
"""


def _rules() -> dict[str, Any]:
    return folders.rules_from(RULES)


def test_rules_are_validated() -> None:
    given = _rules()
    assert [r["match"] for r in given["rules"]] == ["src/", "src/IO/", "src/IO/c.CPP", "src/libgcc/"]
    assert given["rows"][0]["path"] == "src/sdk/rnc/" and given["libraries"] == []
    for bad in (
        '[[rule]]\nmatch = "src/a/"\nto = "src/b"\n',
        '[[rule]]\nmatch = "src/a/"\nto = "src/b/"\n[[rule]]\nmatch = "src/a/"\nto = "src/c/"\n',
        '[[rule]]\nmatch = "src/a/"\nto = "src/b/"\nrow = { kind = "other" }\n',
        '[[row]]\npath = "src/a"\nkind = "dir"\n',
        "libraries = [1]\n",
        "not toml [",
    ):
        with pytest.raises(Refusal) as refusal:
            folders.rules_from(bad)
        assert refusal.value.findings[0].key == "layout.input"


def test_the_longest_match_wins_and_a_directory_keeps_the_rest_of_the_path() -> None:
    rules = _rules()["rules"]
    assert folders.place(rules, "src/IO/a.CPP") == "src/sdk/io/a.CPP"
    assert folders.place(rules, "src/IO/c.CPP") == "src/game/IO/c.CPP"
    assert folders.place(rules, "src/AI/NAV/x.c") == "src/game/AI/NAV/x.c"
    assert folders.place(rules, "README.md") == "README.md"
    assert folders.place([{"match": "src/a.c", "to": "src/b/z.c"}], "src/a.c") == "src/b/z.c"


def test_moves_name_the_files_that_move_and_the_rows_of_the_rules_that_moved_them() -> None:
    mapping, rows = folders.moves(_rules()["rules"], TRACKED)
    assert mapping["src/IO/b.CPP"] == "src/sdk/io/b.CPP" and "README.md" not in mapping
    assert mapping["src/libgcc/g.h"] == "src/sdk/libgcc/g.h"
    assert rows == [{"path": "src/sdk/io/", "kind": "dir", "toolchain": "gcc-sn"}]


def test_two_files_to_one_path_or_onto_a_file_that_stays_are_refused() -> None:
    rules = [{"match": "src/a/", "to": "src/z/"}, {"match": "src/b/", "to": "src/z/"}]
    with pytest.raises(Refusal) as refusal:
        folders.moves(rules, ["src/a/x.c", "src/b/x.c"])
    assert refusal.value.findings[0].key == "layout.collision"
    with pytest.raises(Refusal):
        folders.moves([{"match": "src/a/", "to": "src/b/"}], ["src/a/x.c", "src/b/x.c"])


def test_a_directory_folds_only_when_all_its_files_move_the_same_way() -> None:
    mapping, _ = folders.moves(_rules()["rules"], TRACKED)
    dirs = folders.folded(TRACKED, mapping)
    assert dirs["src/libgcc/"] == "src/sdk/libgcc/"
    assert "src/IO/" not in dirs and "src/" not in dirs


def _rewrite(libraries: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    mapping, rows = folders.moves(_rules()["rules"], TRACKED)
    text, changed = folders.rewrite(CONFIG, TRACKED, mapping, rows, "gcc-def", libraries)
    return tomllib.loads(text), changed


def test_the_config_follows_the_move() -> None:
    document, changed = _rewrite()
    assert document["libraries"] == ["src/sdk/libgcc/"]
    assert document["cppflags"] == ["-DX", "-Isrc/sdk/libgcc"]
    assert document["dir"] == {"src/sdk/io/": {"toolchain": "gcc-sn"}}
    assert document["file"]["src/sdk/libgcc/g.c"]["cflags"] == ["-mfp32", "-Isrc/sdk/libgcc"]
    assert changed["added"] == ["src/sdk/io/"]


def test_a_row_the_new_directory_row_makes_dead_goes_and_a_file_it_would_change_keeps_its_toolchain() -> None:
    document, changed = _rewrite()
    assert "src/sdk/io/a.CPP" not in document["file"] and changed["dropped"] == ["src/sdk/io/a.CPP"]
    assert document["file"]["src/sdk/io/b.CPP"] == {"toolchain": "gcc-def", "cflags": []}
    assert changed["exceptions"] == ["src/sdk/io/b.CPP"]
    assert "src/game/IO/c.CPP" not in document["file"]  # its directory row does not reach it


def test_libraries_come_from_the_rules_when_they_give_them() -> None:
    document, _ = _rewrite([])
    assert "libraries" not in document
    document, _ = _rewrite(["src/sdk/libgcc/"])
    assert document["libraries"] == ["src/sdk/libgcc/"]


def test_a_library_row_without_flags_is_not_dead() -> None:
    mapping, rows = folders.moves(_rules()["rules"], TRACKED)
    rows = [*rows, {"path": "src/sdk/libgcc/", "kind": "library", "origin": "gcc"}]
    text, changed = folders.rewrite(CONFIG, TRACKED, mapping, rows, "gcc-def", [])
    assert tomllib.loads(text)["library"] == {"src/sdk/libgcc/": {"origin": "gcc"}} and changed["dropped"] != [
        "src/sdk/libgcc/"
    ]


def test_a_directory_row_over_files_that_split_is_refused() -> None:
    mapping, rows = folders.moves(_rules()["rules"], TRACKED)
    with pytest.raises(Refusal) as refusal:
        folders.rewrite(CONFIG + '\n[dir."src/IO/"]\ncflags = ["-g"]\n', TRACKED, mapping, rows, "gcc-def")
    assert refusal.value.findings[0].key == "layout.split"


PATCH = """\
diff --git a/src/IO/a.CPP b/src/IO/a.CPP
index 1111111..2222222 100644
--- a/src/IO/a.CPP
+++ b/src/IO/a.CPP
@@ -1,2 +1,2 @@
--- a/src/IO/a.CPP
-old
+new
diff --git a/src/AI/n.c b/src/AI/n.c
new file mode 100644
--- /dev/null
+++ b/src/AI/n.c
@@ -0,0 +1 @@
+x
"""


def test_a_patch_gets_the_new_paths_in_its_headers_only() -> None:
    rules = _rules()["rules"]
    mapping, _ = folders.moves(rules, TRACKED)
    out = folders.carry_patch(PATCH, rules, mapping)
    assert "diff --git a/src/sdk/io/a.CPP b/src/sdk/io/a.CPP" in out
    assert "--- a/src/sdk/io/a.CPP\n+++ b/src/sdk/io/a.CPP\n@@" in out
    assert "\n--- a/src/IO/a.CPP\n-old" in out  # a hunk line that looks like a header is content
    assert "+++ b/src/game/AI/n.c" in out and "--- /dev/null" in out


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return done.stdout


def _rules_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.toml"
    path.write_text(
        '[[rule]]\nmatch = "src/main/"\nto = "src/game/main/"\n[[rule]]\nmatch = "src/ovl/"\nto = "src/lib/ovl/"\n'
        'row = { kind = "library", origin = "an overlay library" }\n'
    )
    return path


def test_plan_writes_the_map_and_moves_nothing(cfg: Config, tmp_path: Path) -> None:
    result = folders.plan(cfg, {"rules": _rules_file(tmp_path)})
    root = cfg.project.root
    assert result["moved"] == 3 and (root / "src/main/c_func.c").is_file()
    lines = (root / folders.MAP).read_text().splitlines()
    assert "src/ovl/fuzzy.c\tsrc/lib/ovl/fuzzy.c" in lines
    document = json.loads((root / folders.PLAN).read_text())
    assert document["rows"] == [{"path": "src/lib/ovl/", "kind": "library", "origin": "an overlay library"}]


def test_plan_without_rules_is_refused(cfg: Config) -> None:
    with pytest.raises(Refusal) as refusal:
        folders.plan(cfg, {})
    assert refusal.value.findings[0].key == "layout.input"


def test_carry_moves_untracked_files_and_writes_the_edits_as_a_patch(cfg: Config, tmp_path: Path) -> None:
    root = cfg.project.root
    folders.plan(cfg, {"rules": _rules_file(tmp_path)})
    (root / "src/main/new.c").write_text("int n;\n")
    (root / "src/main/c_func.c").write_text("s32 c_func(void) {\n    return 2;\n}\n")
    result = folders.carry(cfg, {})
    assert result["moved"] == ["src/game/main/new.c"] and (root / "src/game/main/new.c").is_file()
    assert not (root / "src/main/new.c").exists() and (root / "src/main/c_func.c").is_file()
    patch = Path(result["patch"]).read_text()
    assert "diff --git a/src/game/main/c_func.c b/src/game/main/c_func.c" in patch and "+    return 2;" in patch
    assert _git(root, "status", "--porcelain").count("c_func.c") == 1  # the tracked file itself is not touched


def test_apply_without_a_plan_or_with_a_dirty_file_is_refused(cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(Refusal) as refusal:
        folders.apply(cfg, {})
    assert "no plan" in refusal.value.findings[0].reason
    folders.plan(cfg, {"rules": _rules_file(tmp_path)})
    (cfg.project.root / "src/main/c_func.c").write_text("changed\n")
    with pytest.raises(Refusal) as refusal:
        folders.apply(cfg, {})
    assert "uncommitted" in refusal.value.findings[0].reason


def test_verify_compares_sections_through_the_map(cfg: Config, tmp_path: Path) -> None:
    root = cfg.project.root
    folders.plan(cfg, {"rules": _rules_file(tmp_path)})
    before = tmp_path / "before"
    for version in cfg.project.versions:
        for old, new, same in (
            ("src/main/c_func.c", "src/game/main/c_func.c", True),
            ("src/ovl/fuzzy.c", "src/lib/ovl/fuzzy.c", False),
        ):
            (before / "build" / version / old).parent.mkdir(parents=True, exist_ok=True)
            (before / "build" / version / f"{old}.o").write_bytes(fixture.elf_with_symbols(["a"], []))
            (root / "build" / version / new).parent.mkdir(parents=True, exist_ok=True)
            built = fixture.elf_with_symbols(["a"], [])
            other = built.replace(b"\x03\xe0\x00\x08", b"\x03\xe0\x00\x09")  # jr ra becomes another word
            (root / "build" / version / f"{new}.o").write_bytes(built if same else other)
    (before / f"{cfg.project.name}.ld").write_text("SECTIONS { src/main/c_func.c.o(.text) }\n")
    (root / f"{cfg.project.name}.ld").write_text("SECTIONS { src/game/main/c_func.c.o(.text) }\n")
    result = folders.verify(cfg, {"old": before})
    assert result["same"] == 2 and result["differ"] == 2 and result["link_script"] is True
    assert result["bad"][0].endswith("src/lib/ovl/fuzzy.c")


def test_a_map_file_names_each_file_once_and_a_trailing_slash_keeps_the_name() -> None:
    rules = folders.rules_from_map("# c\nsrc/a.CPP\tsrc/m/A.CPP\nsrc/a.H\tinclude/m/\nsrc/same.c\tsrc/same.c\n")
    assert rules == [{"match": "src/a.CPP", "to": "src/m/A.CPP"}, {"match": "src/a.H", "to": "include/m/"}]
    assert folders.place(rules, "src/a.H") == "include/m/a.H"
    with pytest.raises(Refusal):
        folders.rules_from_map("src/a.c\tx.c\nsrc/a.c\ty.c\n")
    with pytest.raises(Refusal):
        folders.rules_from_map("src/a.c\n")


def test_includes_follow_a_moved_header_and_a_moved_includer() -> None:
    tracked = ["src/a/x.c", "src/a/x.h", "inc/k/y.h", "inc/k/z.h", "inc/k/w.h", "src/b/u.c"]
    texts = {
        "src/a/x.c": '#include "x.h"\n#include "k/y.h"\n#include <stdio.h>\n',
        "src/b/u.c": '#include "k/y.h"\n#include "k/z.h"\n',
        "inc/k/w.h": '#include "y.h"\n',
    }
    mapping = {"src/a/x.h": "src/m/x.h", "inc/k/y.h": "inc/m/y.h", "src/a/x.c": "src/m/x.c"}
    out = folders.includes(texts, tracked, mapping)
    assert out["src/a/x.c"] == '#include "x.h"\n#include "m/y.h"\n#include <stdio.h>\n'
    assert out["src/b/u.c"] == '#include "m/y.h"\n#include "k/z.h"\n'
    assert out["inc/k/w.h"] == '#include "../m/y.h"\n'


def test_plan_takes_a_map_without_a_rules_file(cfg: Config, tmp_path: Path) -> None:
    mapped = tmp_path / "m.tsv"
    mapped.write_text("src/main/c_func.c\tsrc/main/renamed.c\n")
    result = folders.plan(cfg, {"map": mapped})
    assert result["moved"] == 1 and (cfg.project.root / "src/main/c_func.c").is_file()


def test_makefile_flag_rules_follow_a_moved_file_and_a_folded_directory() -> None:
    tracked = ["src/IO/a.CPP", "src/IO/b.CPP", "src/game/main.c", "src/stay/s.c"]
    mapping = {"src/IO/a.CPP": "src/sdk/io/a.CPP", "src/IO/b.CPP": "src/sdk/io/b.CPP", "src/game/main.c": "src/g/m.c"}
    rule = "$(foreach v,$(VERSIONS),build/{tree}$(v)/{path}): TC := gcc-sn\n"
    text = "".join(rule.format(tree=t, path=p) for t in ("", "nmraw/") for p in ("src/game/main.c.o", "src/IO/%.o"))
    text += rule.format(tree="", path="src/stay/s.c.o") + "OBJS := build/$(v)/src/game/main.c.o\n"
    out, count = folders.makefile_rules(text, tracked, mapping)
    assert count == 4
    assert out.count("build/$(v)/src/g/m.c.o): TC") == 1 and out.count("build/nmraw/$(v)/src/g/m.c.o): TC") == 1
    assert out.count("build/$(v)/src/sdk/io/%.o): TC") == 1 and out.count("build/nmraw/$(v)/src/sdk/io/%.o): TC") == 1
    assert "src/stay/s.c.o): TC" in out and "OBJS := build/$(v)/src/game/main.c.o\n" in out
    assert folders.makefile_rules(text, tracked, {})[1] == 0


@pytest.mark.parametrize("kind", ["dir", "library", "flat"])
def test_in_place_child_renames_preserve_boundary_facts(kind: str) -> None:
    tracked = ["src/sdk/a.CPP", "src/sdk/b.c", "src/sdk/header.h", "src/game/g.c"]
    mapping = {"src/sdk/a.CPP": "src/sdk/named.CPP"}
    facts = 'toolchain = "gcc-sdk"\ncflags = ["-g", "-Isrc/sdk"]\nomit = ["eu"]\n'
    text = 'schema = 4\nname = "T"\ntoolchain = "gcc-default"\n'
    if kind == "flat":
        text += 'libraries = ["src/sdk/"]\n[dir."src/sdk/"]\n' + facts
    else:
        text += f'[{kind}."src/sdk/"]\n' + facts
        if kind == "library":
            text += 'origin = "sdk-source"\n'
    result, changes = folders.rewrite(text, tracked, mapping, [], "gcc-default")
    assert tomllib.loads(result) == tomllib.loads(text)
    assert changes["exceptions"] == []
    for move in ({"src/sdk/a.CPP": "src/game/a.CPP"}, {"src/game/g.c": "src/sdk/g.c"}):
        with pytest.raises(Refusal, match="different places"):
            folders.rewrite(text, tracked, move, [], "gcc-default")
    assert folders.rewrite(result, [mapping.get(p, p) for p in tracked], {}, [], "gcc-default")[0] == result
