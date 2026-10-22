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
