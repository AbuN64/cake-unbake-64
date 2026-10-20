"""Landing by replacing the stand-in line in the owning file; tree, make, policy and measure are stubbed."""

from __future__ import annotations

import hashlib
import re
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


COMPOUND = (
    '#include "game.h"\nint keep(void) { return 1; }\n'
    '#if defined(VERSION_US) && defined(NON_MATCHING)\n#include "types.h"\nint func_80000010(void) { old(); }\n#else\n'
    '#if !defined(VERSION_EU)\nINCLUDE_ASM("main", func_80000010);\n#endif\n#endif\nINCLUDE_ASM("main", other);\n'
)
WHOLE_FILE = (
    '#include "game.h"\nint keep(void) { return 1; }\n'
    '#if defined(VERSION_US) && defined(NON_MATCHING)\n#include "types.h"\n'
    "int func_80000010(void) {\n    return 2;\n}\n"
    '#else\n#if !defined(VERSION_EU)\nINCLUDE_ASM("main", func_80000010);\n#endif\n#endif\n'
)


@pytest.mark.parametrize(
    "guard",
    [
        "#ifdef NON_MATCHING",
        "#  ifdef   NON_MATCHING  ",
        "#if defined(NON_MATCHING)",
        "#if defined NON_MATCHING",
        "#if defined(VERSION_US) && defined(NON_MATCHING)",
        "#if (defined(NON_MATCHING) || 0) && (1 || 0) /* note */",
    ],
)
def test_every_spelling_of_the_non_matching_guard_is_one(guard):
    assert tree.guard(guard + "\n") is not None
    assert tree.fuzzy_only(
        f'{guard}\nint f(void) {{ return 1; }}\n#else\nINCLUDE_ASM("m", f);\n#endif\n', []
    ).startswith("#")
    assert tree.non_matching_blocks([guard + "\n", "x\n", "#else\n", "y\n", "#endif\n"]) == [(0, 2, 4)]


@pytest.mark.parametrize(
    "guard",
    [
        "#if 1",
        "#ifndef NON_MATCHING",
        "#if !defined(NON_MATCHING)",
        "#ifdef OTHER",
        "#if defined(A)",
        "#if FOO(NON_MATCHING)",
    ],
)
def test_other_conditionals_are_not_the_non_matching_guard(guard):
    assert tree.guard(guard + "\n") is None


def test_the_candidate_replaces_the_body_behind_a_compound_guard():
    text = publish.splice(COMPOUND, "main", "func_80000010", CANDIDATE, fuzzy=False)
    assert "old()" not in text and "return a + table[0];" in text and 'INCLUDE_ASM("main", func_80000010)' not in text
    assert 'INCLUDE_ASM("main", other);' in text
    assert text.count("#if") == text.count("#endif")


def test_a_whole_file_candidate_with_a_compound_guard_lands_without_stray_directives():
    text = publish.splice(COMPOUND, "main", "func_80000010", WHOLE_FILE, fuzzy=False)
    assert "return 2;" in text and "old()" not in text and 'INCLUDE_ASM("main", other)' in text
    assert text.count('INCLUDE_ASM("main", func_80000010)') == 0
    assert text.count("#if") == text.count("#endif") and "#else" not in text
    assert text.count('#include "types.h"') == 1 and text.count('#include "game.h"') == 1
    assert [p.name for p in publish.items(text) if p.kind == "function"] == ["keep", "func_80000010"]


def test_a_whole_file_candidate_over_a_plain_stand_in_drops_its_own_wrapper():
    text = publish.splice(OWNER, "main", "func_80000010", WHOLE_FILE, fuzzy=False)
    assert "#if" not in text and "#else" not in text and "return 2;" in text
    assert text.count('INCLUDE_ASM("main", other)') == 1


def test_a_nested_guard_and_a_plain_ifdef_whole_file_candidate_still_land():
    whole = (
        "#ifdef NON_MATCHING\nint func_80000010(void) { return 3; }\n#else\n"
        'INCLUDE_ASM("main", func_80000010);\n#endif\n'
    )
    for owner in (NESTED, COMPOUND):
        text = publish.splice(owner, "main", "func_80000010", whole, fuzzy=False)
        assert "return 3;" in text and text.count("#if") == text.count("#endif")
        assert 'INCLUDE_ASM("main", func_80000010)' not in text


def test_a_fuzzy_candidate_over_a_compound_guard_keeps_one_guarded_block():
    text = publish.splice(COMPOUND, "main", "func_80000010", WHOLE_FILE, fuzzy=True)
    assert "old()" not in text and text.count("#ifdef NON_MATCHING") == 1
    assert text.index("return 2;") < text.index("#else") < text.index('INCLUDE_ASM("main", func_80000010)')
    assert text.count("#if") == text.count("#endif")


LATER = (
    "#ifdef NON_MATCHING\nint later(void) {\n    return 4;\n}\n#else\n"
    'INCLUDE_ASM("main", later);\n#endif\nINCLUDE_ASM("main", tail);\n'
)


def test_a_whole_file_candidate_does_not_bring_the_files_later_functions_again():
    owner = COMPOUND + LATER
    source = owner.replace("int func_80000010(void) { old(); }", "int func_80000010(void) {\n    return 5;\n}")
    text = publish.splice(owner, "main", "func_80000010", source, fuzzy=False)
    assert "return 5;" in text and "old()" not in text
    assert text.count("int later(void)") == 1 and text.count('INCLUDE_ASM("main", later)') == 1
    assert text.count('INCLUDE_ASM("main", tail)') == 1 and text.count("#if") == text.count("#endif")


def test_a_helper_after_the_function_that_the_file_lacks_still_comes_along():
    source = CANDIDATE + "static int helper(void) {\n    return 6;\n}\n"
    text = publish.splice(OWNER, "main", "func_80000010", source, fuzzy=False)
    assert text.count("static int helper(void)") == 1


# ---------------------------------------------------------------- what a candidate brings, each conditional whole
LANDED = (
    '#include "game.h"\n#if !defined(NON_MATCHING) || defined(VERSION_EU)\nINCLUDE_ASM("main", before);\n#endif\n'
    "extern int count;\n/* sets the count */\nint landed(void) {\n    count = 1;\n    return 0;\n}\n"
    '#ifndef NON_MATCHING\nINCLUDE_ASM("main", later);\n#endif\n'
    "#ifdef NON_MATCHING\nint later(void) {\n    return 2;\n}\n#if !defined(VERSION_EU)\nint tail(void) {\n"
    "    return 3;\n}\n#endif\n#endif\n"
)


def test_a_whole_file_candidate_of_a_landed_function_replaces_only_its_definition():
    source = LANDED.replace("count = 1;", "count = 2;")
    text, origins = publish.replace_traced(LANDED, "landed", source)
    assert text == source and len(origins) == len(source.splitlines())
    assert origins[source.splitlines().index("int landed(void) {")] == ("candidate", 7)


def test_a_landed_function_without_its_file_still_brings_what_the_file_lacks():
    source = "extern int count;\nextern int other;\nint landed(void) {\n    return other;\n}\n"
    text, _ = publish.replace_traced(LANDED, "landed", source)
    assert text.count("extern int count;") == 1 and text.count("extern int other;") == 1
    assert text.index("extern int other;") < text.index("int landed(void)") and "count = 1;" not in text
    assert "/* sets the count */" in text  # a candidate without its own comment keeps the file's


SHARED = (
    '#include "game.h"\n#if !defined(NON_MATCHING) || defined(VERSION_EU)\nINCLUDE_ASM("main", before);\n#endif\n'
    '#ifndef NON_MATCHING\nINCLUDE_ASM("main", func_80000010);\n#endif\n'
    '#ifndef NON_MATCHING\nINCLUDE_ASM("main", later);\n#endif\n'
    '#ifdef NON_MATCHING\n#include "types.h"\nint func_80000010(void) {\n    return 1;\n}\n'
    "int later(void) {\n    return 2;\n}\n"
    "#if !defined(VERSION_EU)\nint tail(void) {\n    return 3;\n}\n#endif\n#endif\n"
)
SHARED_SEED = (  # a resume seed: the file's includes and the whole NON_MATCHING block the function shares
    '#include "game.h"\n#include "types.h"\n#if defined(VERSION_US)\nextern int us_only;\n#endif\n'
    "int func_80000010(void) {\n    return 5;\n}\n"
    "int later(void) {\n    return 2;\n}\n#if !defined(VERSION_EU)\nint tail(void) {\n    return 3;\n}\n#endif\n"
)


def test_a_candidate_conditional_comes_whole_though_the_file_already_has_an_endif():
    text = publish.splice(SHARED, "main", "func_80000010", SHARED_SEED, fuzzy=False)
    assert len(re.findall(r"^#\s*if", text, re.M)) == len(re.findall(r"^#\s*endif", text, re.M))
    stand_in = text.index("return 5;")
    assert text.rindex("#if defined(VERSION_US)\nextern int us_only;\n#endif\n", 0, stand_in) > 0
    # the stand-in is kept apart from its draft: the block's include is made live above the body, not brought twice
    assert text.count('#include "types.h"') == 1 and text.index('#include "types.h"') < stand_in
    # the functions after it that the file holds behind its NON_MATCHING guard are the file's, not brought again
    assert text.count("int later(void)") == 1 and text.count("int tail(void)") == 1


def test_a_whole_conditional_already_visible_before_the_stand_in_is_not_brought_again():
    source = '#if !defined(NON_MATCHING) || defined(VERSION_EU)\nINCLUDE_ASM("main", before);\n#endif\n' + (
        "int func_80000010(void) {\n    return 5;\n}\n"
    )
    text = publish.splice(SHARED, "main", "func_80000010", source, fuzzy=False)
    assert text.count('INCLUDE_ASM("main", before);') == 1
    assert len(re.findall(r"^#\s*if", text, re.M)) == len(re.findall(r"^#\s*endif", text, re.M))


def test_only_the_conditionals_in_a_branch_still_open_are_visible():
    before = (
        "#if A\nint a;\n#endif\n#ifdef NON_MATCHING\n#if B\nint b;\n#endif\n#else\n#if C\nint c;\n#endif\n"
        "#if D\n#if E\nint e;\n#endif\n"  # D is still open: E sits in its open branch
    )
    seen = publish._visible_groups(before)
    assert seen == {"#if A int a; #endif", "#if C int c; #endif", "#if E int e; #endif"}  # B sits in a closed branch


# ---------------------------------------------------------------- a stand-in kept apart from its draft
APART = (
    '#include "game.h"\n'
    '#ifndef NON_MATCHING\nINCLUDE_ASM("main", f);\n#endif\n\n'
    '#ifndef NON_MATCHING\nINCLUDE_ASM("main", later);\n#endif\n\n'
    '#ifdef NON_MATCHING\n#include "hud.h"\ntypedef struct Hud { int x; } Hud;\n\n/* the screen */\nextern Hud *hud;\n'
    "int helper(int);\n\n/* old draft */\nint f(void) {\n    return hud->x;\n}\n\n"
    "int later(void) {\n    return 2;\n}\n#endif\n"
)
CAND_F = "/* reads the hud */\nint f(void) {\n    return hud->x + 1;\n}\n"


def test_landing_a_stand_in_kept_apart_from_its_draft_makes_the_body_and_its_types_live_in_rom_order():
    text, origins = publish.splice_traced(APART, "main", "f", CAND_F, fuzzy=False)
    assert len(origins) == len(text.splitlines())
    assert 'INCLUDE_ASM("main", f);' not in text and "old draft" not in text and "return hud->x;" not in text
    live = text[: text.index("#ifdef NON_MATCHING")]  # all of it builds without NON_MATCHING
    assert live.index('#include "hud.h"') < live.index("typedef struct Hud") < live.index("/* the screen */")
    assert live.index("int helper(int);") < live.index("/* reads the hud */") < live.index('INCLUDE_ASM("main", later)')
    assert live.count("#ifndef NON_MATCHING") == 1  # the landed stand-in's guard is gone, the later one's stays
    assert text.count("typedef struct Hud") == 1 and "int later(void) {\n    return 2;\n}\n#endif\n" in text
    assert origins[text.splitlines().index("    return hud->x + 1;")][0] == "candidate"


def test_landing_the_last_draft_of_its_block_drops_the_empty_block():
    text = publish.splice(APART.replace("int later(void) {\n    return 2;\n}\n", ""), "main", "f", CAND_F, False)
    assert "#ifdef NON_MATCHING" not in text and "return hud->x + 1;" in text


def test_a_stand_in_sharing_its_guard_has_the_guard_closed_around_the_landed_body():
    one = 'INCLUDE_ASM("main", f);\n#endif\n\n#ifndef NON_MATCHING\nINCLUDE_ASM("main", later);\n'
    for both, want in (
        ('INCLUDE_ASM("main", f);\nINCLUDE_ASM("main", later);\n', "f first"),
        ('INCLUDE_ASM("main", early);\nINCLUDE_ASM("main", f);\nINCLUDE_ASM("main", later);\n', "f between"),
    ):
        text = publish.splice(APART.replace(one, both), "main", "f", CAND_F, fuzzy=False)
        head = text[: text.index("#ifdef NON_MATCHING")]
        assert "#ifndef NON_MATCHING\n#endif" not in head, want
        assert head.index("return hud->x + 1;") < head.rindex("#ifndef NON_MATCHING") < head.index('("main", later)')
        assert head.count("#ifndef NON_MATCHING") == head.count("#endif") == (2 if "early" in both else 1), want


def test_a_stand_in_after_its_drafts_block_gets_the_types_before_the_block():
    after = (
        '#include "game.h"\n#ifdef NON_MATCHING\ntypedef struct Hud { int x; } Hud;\nextern Hud *hud;\n'
        "int f(void) {\n    return 0;\n}\nint later(void) {\n    return 2;\n}\n#endif\n"
        '#ifndef NON_MATCHING\nINCLUDE_ASM("main", later);\nINCLUDE_ASM("main", f);\n#endif\n'
    )
    text = publish.splice(after, "main", "f", CAND_F, fuzzy=False)
    assert text.index("typedef struct Hud") < text.index("#ifdef NON_MATCHING") < text.index("return hud->x + 1;")
    assert text.index('INCLUDE_ASM("main", later)') < text.index("return hud->x + 1;")


def test_types_after_a_draft_that_gives_bytes_stay_in_the_block():
    held = APART.replace('#include "hud.h"\n', '#include "hud.h"\nint early(void) {\n    return 1;\n}\n')
    text = publish.splice(held, "main", "f", CAND_F, fuzzy=False)
    block = text[text.index("#ifdef NON_MATCHING") :]
    assert "int early(void)" in block and "typedef struct Hud" in block and '#include "hud.h"' not in block


def test_a_draft_block_with_another_side_is_refused_by_name():
    other = APART.replace(
        "    return 2;\n}\n#endif\n", '    return 2;\n}\n#else\nINCLUDE_ASM("main", later2);\n#endif\n'
    )
    assert publish.measured_apart(other, "main", "f")
    with pytest.raises(Refusal) as refused:
        publish.splice(other, "main", "f", CAND_F, fuzzy=False)
    assert "no #else" in refused.value.findings[0].reason
