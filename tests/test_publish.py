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


def test_holders_name_the_segment_of_the_name_not_of_the_file(world):
    held, path, segment = publish.holders(world.tree, "other")
    assert (held, path, segment) == (("eu", "us"), "src/a.c", "ovl")  # the file's own segment is main
    assert publish.holders(world.tree, "func_80000010")[2] == "main"


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
