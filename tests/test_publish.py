"""Landing by replacing the stand-in line in the owning file; tree, make, policy and measure are stubbed."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from unbake import journal, publish, tree
from unbake.contracts import Refusal, Submission

OWNER = (
    '#include "game.h"\n\nint keep(void) { return 1; }\n\nINCLUDE_ASM("main", func_80000010);\n\n'
    'INCLUDE_ASM("main", other);\n'
)
CANDIDATE = (
    '#include "game.h"\nstatic int table[2];\n\n/* adds */\nint func_80000010(int a) {\n    return a + table[0];\n}\n'
)


def rom(version: str) -> bytes:
    return f"rom-{version}".encode()


class World:
    """A project on disk with a fake tree, make and policy."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, versions=("us", "eu"), held=("us", "eu")):
        self.root = tmp_path
        (tmp_path / "src").mkdir()
        (tmp_path / "src/a.c").write_text(OWNER)
        (tmp_path / "cand.c").write_text(CANDIDATE)
        facts = {v: SimpleNamespace(sha1=hashlib.sha1(rom(v)).hexdigest()) for v in versions}
        self.config = SimpleNamespace(
            project=SimpleNamespace(root=tmp_path, name="game", version=facts), host=SimpleNamespace(slots=4, workers=4)
        )
        module = SimpleNamespace(
            path="src/a.c",
            segment="main",
            owns={v: ("keep", "func_80000010", "other") for v in held},
            placed={v: {"other": "ovl"} for v in held},
        )
        self.tree = SimpleNamespace(digest="d", modules=(module,), owner=lambda v, n: "src/a.c" if v in held else None)
        self.makes: list[list[str]] = []
        self.broken: set[str] = set()
        self.scores = {v: 0.9 for v in held}
        self.before = {v: 0.5 for v in held}
        monkeypatch.setattr(publish, "tree", SimpleNamespace(capture=lambda config: self.tree))
        monkeypatch.setattr(publish, "process", SimpleNamespace(make=self.make))
        monkeypatch.setattr(publish, "policy", SimpleNamespace(evaluate=lambda *a: (), scope=lambda *a: ((), ())))
        monkeypatch.setattr(publish, "api", SimpleNamespace(snapshot=lambda c: None, score=lambda c, n: self.before))
        monkeypatch.setattr(publish, "native", SimpleNamespace(measure=self.measure))
        self.monkeypatch = monkeypatch

    def make(self, config, targets, jobs=0):
        self.makes.append(list(targets))
        for target in targets:
            version = target.split("/")[1]
            path = self.root / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wrong" if version in self.broken else rom(version))

    def measure(self, config, version, name, path, text):
        return SimpleNamespace(score=self.scores[version], symptoms={}, missing=())

    def submission(self, operation="publish") -> Submission:
        return Submission(
            "id", operation, "func_80000010", None, "cand.c", "x", {"add": [], "omit": []}, "h", (), "submit", "", "i",
            {}, {}, (),
        )  # fmt: skip

    @property
    def owner(self) -> str:
        return (self.root / "src/a.c").read_text()


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_items_split_functions_directives_and_declarations():
    parts = publish.items('#include "x.h"\nstatic int a[2] = {1, 2};\n/* c */\nint f(int x) {\n  return "}" [0];\n}\n')
    assert [(p.kind, p.name) for p in parts] == [("directive", ""), ("decl", ""), ("comment", ""), ("function", "f")]


def test_candidate_splits_the_definition_from_its_declarations():
    declared, body = publish.candidate(CANDIDATE, "func_80000010")
    assert body.startswith("/* adds */\nint func_80000010") and body.endswith("}\n")
    assert declared == '#include "game.h"\nstatic int table[2];'


def test_candidate_refuses_a_missing_definition():
    with pytest.raises(Refusal) as error:
        publish.candidate(CANDIDATE, "nothing")
    assert error.value.findings[0].key == "land.request"


def test_candidate_refuses_a_conditional_around_the_function():
    with pytest.raises(Refusal):
        publish.candidate("#if VERSION_US\nint f(void) { return 1; }\n", "f")


def test_splice_replaces_the_line_and_adds_declarations_right_above_the_candidate():
    text = publish.splice(OWNER, "main", "func_80000010", CANDIDATE, fuzzy=False)
    assert 'INCLUDE_ASM("main", func_80000010)' not in text
    assert 'INCLUDE_ASM("main", other);' in text
    assert text.count('#include "game.h"') == 1  # already there: not added twice
    assert text.index("int keep(void)") < text.index("static int table[2];") < text.index("int func_80000010")


def test_declarations_land_after_an_include_the_file_makes_between_its_stand_ins():
    owner = 'INCLUDE_ASM("main", first);\n#include "types.h"\nINCLUDE_ASM("main", func_80000010);\n'
    text = publish.splice(
        owner,
        "main",
        "func_80000010",
        '#include "types.h"\ns32 g(void);\nint func_80000010(int a) {\n    return g();\n}\n',
        fuzzy=False,
    )
    assert text.index('#include "types.h"') < text.index("s32 g(void);") < text.index("int func_80000010")
    assert text.count('#include "types.h"') == 1


def test_splice_fuzzy_wraps_the_body_over_the_stand_in():
    text = publish.splice(OWNER, "main", "func_80000010", CANDIDATE, fuzzy=True)
    assert "#ifdef NON_MATCHING\n/* adds */\nint func_80000010(int a) {" in text
    assert '}\n#else\nINCLUDE_ASM("main", func_80000010);\n#endif\n' in text


def test_splice_replaces_an_earlier_fuzzy_block_by_an_exact_body():
    fuzzy = publish.splice(OWNER, "main", "func_80000010", CANDIDATE, fuzzy=True)
    exact = publish.splice(fuzzy, "main", "func_80000010", CANDIDATE, fuzzy=False)
    assert "NON_MATCHING" not in exact and "#else" not in exact


def test_splice_without_a_stand_in_is_a_duplicate():
    landed = publish.splice(OWNER, "main", "func_80000010", CANDIDATE, fuzzy=False)
    with pytest.raises(Refusal) as error:
        publish.splice(landed, "main", "func_80000010", CANDIDATE, fuzzy=False)
    assert error.value.findings[0].key == "land.duplicate"


def test_an_exact_land_replaces_the_line(world):
    receipt = publish.land(world.config, world.submission())
    assert receipt.files == ("src/a.c",) and receipt.operation == "publish"
    assert 'INCLUDE_ASM("main", func_80000010)' not in world.owner
    assert "return a + table[0];" in world.owner
    assert world.makes == [["build/eu/game.z64", "build/us/game.z64"]]


def test_a_fuzzy_land_writes_the_non_matching_block(world):
    receipt = publish.land(world.config, world.submission("fuzzy"))
    assert receipt.operation == "fuzzy" and receipt.message == "fuzzy func_80000010 90.0%"
    assert '#else\nINCLUDE_ASM("main", func_80000010);\n#endif' in world.owner
    assert world.makes  # the stand-in side still builds to the base ROM


def test_no_gain_is_refused_and_nothing_is_built(world):
    world.scores = {"us": 0.505, "eu": 0.9}  # the minimum rises 0.005 < min_gain
    with pytest.raises(Refusal) as error:
        publish.land(world.config, world.submission("fuzzy"))
    assert error.value.findings[0].key == "fuzzy.no_gain"
    assert world.owner == OWNER and world.makes == []


def test_a_sha1_mismatch_is_refused_with_the_files_untouched(world):
    world.broken = {"eu"}
    with pytest.raises(Refusal) as error:
        publish.land(world.config, world.submission())
    finding = error.value.findings[0]
    assert finding.key == "land.not_exact" and finding.versions == ("eu",)
    assert world.owner == OWNER
    assert not (world.root / ".unbake/journal.json").exists()


def test_the_write_goes_through_the_journal(world, monkeypatch):
    applied = []
    real = journal.apply
    monkeypatch.setattr(journal, "apply", lambda c, p, r: applied.append(p.operation) or real(c, p, r))
    publish.land(world.config, world.submission())
    assert applied == ["publish"]


def test_owners_name_the_segment_of_the_name_not_of_the_file(world):
    (owner,) = publish.owners(world.tree, "other")
    assert owner == publish.Owner("src/a.c", "ovl", ("eu", "us"))  # the file's own segment is main
    assert publish.owners(world.tree, "func_80000010")[0].segment == "main"


HELPER = (
    "static double twice(double value)\n{\n    return value + value;\n}\n"
    "double func_80000010(double value)\n{\n    return twice(value);\n}\n"
)


@pytest.mark.parametrize(
    "helper",
    [
        "static double twice(double value) { return value + value; }",
        "static double twice(double value)\n{\n    return value + value;\n}",
    ],
)
def test_a_helpers_closing_brace_survives_whatever_the_brace_layout(helper):
    source = helper + "\ndouble func_80000010(double value)\n{\n    return twice(value);\n}\n"
    text = publish.splice(OWNER, "main", "func_80000010", source, fuzzy=False)
    assert text.count("{") == text.count("}")
    assert [p.name for p in publish.items(text) if p.kind == "function"] == ["keep", "twice", "func_80000010"]
    assert "func_80000010" in tree._scan(text)[0]


NESTED = (
    '#include "game.h"\n#ifdef NON_MATCHING\nvoid func_80000010(void) { old(); }\n#else\n'
    '#if !defined(VERSION_EU)\nINCLUDE_ASM("main", func_80000010);\n#endif\n'
    '#if defined(VERSION_EU)\nINCLUDE_ASM("main", other);\n#endif\n#endif\n'
)


def test_the_candidate_replaces_a_non_matching_body_and_keeps_the_nested_version_guards():
    text = publish.splice(NESTED, "main", "func_80000010", CANDIDATE, fuzzy=False)
    assert "old()" not in text and "NON_MATCHING" not in text
    assert "return a + table[0];" in text and 'INCLUDE_ASM("main", func_80000010)' not in text
    assert '#if defined(VERSION_EU)\nINCLUDE_ASM("main", other);\n#endif' in text
    assert text.count("#if") == text.count("#endif")


def test_a_fuzzy_candidate_replaces_the_non_matching_body_over_the_untouched_else_side():
    text = publish.splice(NESTED, "main", "func_80000010", CANDIDATE, fuzzy=True)
    assert "old()" not in text and text.count("#ifdef NON_MATCHING") == 1
    assert text.index("return a + table[0];") < text.index("#else") < text.index('INCLUDE_ASM("main", func_80000010)')
    assert '#if defined(VERSION_EU)\nINCLUDE_ASM("main", other);' in text


def test_a_declaration_is_reused_only_when_visible_before_the_candidate_in_its_branch():
    owner = (
        '#include "game.h"\n#ifdef NON_MATCHING\nint func_80000010(void) { return 0; }\n#else\n'
        'INCLUDE_ASM("main", func_80000010);\n#endif\n'
        '#ifdef NON_MATCHING\n#include "types.h"\nint later(void);\n#endif\n'
        '#ifdef OTHER\n#include "types.h"\n#endif\n'
    )
    source = '#include "types.h"\n#include "game.h"\nint func_80000010(void) {\n    return 1;\n}\n'
    text = publish.splice(owner, "main", "func_80000010", source, fuzzy=False)
    assert text.index('#include "types.h"') < text.index("int func_80000010")  # not borrowed from a later block
    assert text.count('#include "game.h"') == 1  # visible before, unconditional: reused
    assert text.count('#include "types.h"') == 3


def _split_tree():
    """func_x owned by a.c in us and us-rev1, by b.c in eu."""
    owners = {("us", "func_x"): "a.c", ("us-rev1", "func_x"): "a.c", ("eu", "func_x"): "b.c"}
    modules = tuple(
        SimpleNamespace(
            path=p, segment="main", placed={}, owns={v: ("func_x",) for (v, _), o in owners.items() if o == p}
        )
        for p in ("a.c", "b.c")
    )
    return SimpleNamespace(modules=modules, owner=lambda v, n: owners.get((v, n)))


def test_each_version_is_owned_by_the_file_that_owns_it_there_unless_the_request_names_a_version():
    captured = _split_tree()
    assert publish.owners(captured, "func_x") == (
        publish.Owner("b.c", "main", ("eu",)),
        publish.Owner("a.c", "main", ("us", "us-rev1")),
    )
    assert publish.owners(captured, "func_x", "eu") == (publish.Owner("b.c", "main", ("eu",)),)
    assert publish.owners(captured, "func_x", "us") == (publish.Owner("a.c", "main", ("us", "us-rev1")),)


def test_a_candidate_lands_in_every_file_that_owns_its_name_and_builds_every_version(world):
    (world.root / "src/b.c").write_text('#include "game.h"\nINCLUDE_ASM("main", func_80000010);\n')
    other = SimpleNamespace(path="src/b.c", segment="main", owns={"jp": ("func_80000010",)}, placed={})
    world.tree.modules[0].owns["us"] = ("keep", "func_80000010", "other")
    world.tree.modules = (*world.tree.modules, other)
    owners = {"jp": "src/b.c", "us": "src/a.c", "eu": "src/a.c"}
    world.tree.owner = lambda v, n: owners.get(v)
    world.config.project.version["jp"] = SimpleNamespace(sha1=hashlib.sha1(rom("jp")).hexdigest())
    receipt = publish.land(world.config, world.submission())
    assert sorted(receipt.files) == ["src/a.c", "src/b.c"]
    assert "return a + table[0];" in (world.root / "src/a.c").read_text()
    assert "return a + table[0];" in (world.root / "src/b.c").read_text()
    assert sorted(t.split("/")[1] for t in world.makes[0]) == ["eu", "jp", "us"]


def test_the_new_text_says_which_lines_are_the_files_and_which_the_candidates():
    text, origins = publish.splice_traced(OWNER, "main", "func_80000010", CANDIDATE, fuzzy=False)
    lines = text.splitlines()
    assert len(origins) == len(lines)
    at = lines.index("int func_80000010(int a) {")
    assert origins[at] == ("candidate", 5)  # line 5 of CANDIDATE
    assert origins[lines.index("static int table[2];")] == ("candidate", 2)
    assert origins[lines.index("int keep(void) { return 1; }")] == ("disk", 3)
    assert origins[lines.index('INCLUDE_ASM("main", other);')] == ("disk", 7)  # shifted by what was added above it


DATA_OWNER = '#include "types.h"\n/* data */\nunsigned int D_blob_1[1] = {\n    0x00000000,\n};\nint after;\n'


def test_data_the_file_defines_in_c_is_replaced_by_the_candidate_when_it_has_no_stand_in():
    text = publish.splice(DATA_OWNER, "main", "D_blob_1", "u8 *D_blob_1 = 0;\n", False, "data")
    assert text == '#include "types.h"\n/* data */\nu8 *D_blob_1 = 0;\nint after;\n'


def test_data_without_a_stand_in_or_a_single_definition_is_still_refused():
    with pytest.raises(Refusal) as error:
        publish.splice(DATA_OWNER, "main", "D_other", "int D_other = 1;\n", False, "data")
    assert error.value.findings[0].key == "land.request"
    twice = DATA_OWNER + "unsigned int D_blob_1[1] = {0};\n"
    with pytest.raises(Refusal) as error:
        publish.splice(twice, "main", "D_blob_1", "u8 *D_blob_1 = 0;\n", False, "data")
    assert error.value.findings[0].key == "land.duplicate"
