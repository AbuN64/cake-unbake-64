"""`unbake census` and `unbake unify` over a toy tree: the seams (units, text, make) are fakes, no build."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from unbake import typecensus, typeplan, typeunify
from unbake.contracts import Host, Refusal
from unbake.typecensus import Norm
from unbake.typeunify import Seams, Unit

LIKE = lambda f: f.startswith(("src/", "include/"))  # noqa: E731


def marked(root: Path, unit: str) -> str:
    """What a preprocessor makes of the unit: its header's text under a line marker, then its own."""
    head = (root / "include/h.H").read_text() if (root / "include/h.H").exists() else ""
    return f'# 1 "include/h.H"\n{head}# 1 "{unit}"\n{(root / unit).read_text()}'


class World:
    """A tree of one header and some sources. `fail` names the declarations that make a unit fail to build."""

    def __init__(self, root: Path, files: dict[str, str], units: Sequence[str], fail: Sequence[str] = ()) -> None:
        self.root, self.units, self.fail, self.calls = root, list(units), list(fail), []
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        self.seams = Seams(self.found, self.read, self.make)

    def found(self) -> dict[str, list[str]]:
        return {"us": list(self.units), "de": list(self.units)}

    def read(self, items: Sequence[Unit]) -> list[typeunify.Facts]:
        return [typeunify.facts(marked(self.root, u), u, LIKE) for _, u in items]

    def make(self, targets: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(targets))
        log, code = "make\n", 0
        for t in targets:
            version, unit = re.match(r"build/(\w+)/(.*)\.o$", t).groups()  # type: ignore[union-attr]
            text = marked(self.root, unit)
            if any(bad in text for bad in self.fail):
                log, code = log + f"make: *** [Makefile:1: {t}] Error 1\n", 2
            else:
                out = self.root / t
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    re.sub(r"extern[^\n]*\n", "", (self.root / unit).read_text()) + version
                )  # no declarations
        return code, log


def tree(tmp_path: Path, fail: Sequence[str] = ()) -> World:
    files = {
        "include/h.H": "extern int g;\nextern int h;\n",
        "src/a.CPP": "extern long g;\nint f(void) { return g; }\n",
        "src/b.CPP": "extern short h;\nint k(void) { return h + g; }\n",
    }
    return World(tmp_path, files, ["src/a.CPP", "src/b.CPP"], fail)


# ---------------------------------------------------------------- census parse
def test_statements_are_typed_and_normalised() -> None:
    text = (
        '# 1 "include/t.H"\ntypedef unsigned char u8;\ntypedef struct Foo { int a; } Foo;\n'
        "extern u8 *p[4];\nextern u8 *p[];\nvoid f(Foo *x, int n);\nint g = 3;\nint (*cb)(int a, Foo *b);\n"
        "int body(int a) { return a; }\n"
    )
    typedefs, stmts, files, _ = typeunify.facts(text, "src/u.CPP", LIKE)
    norm = Norm()
    for t in typedefs:
        norm.define(t)
    found = {n: (d[1], d[2]) for n, d in typecensus.declarations(stmts, norm, LIKE)}
    assert found["p"] == ("ext", "unsigned char*[]")
    assert found["f"] == ("proto", "void(struct Foo*,int)")
    assert found["g"] == ("vdef", "int")
    assert found["cb"] == ("ext", "int(*)(int,struct Foo*)") or found["cb"][1] == "int(*)(int,struct Foo*)"
    assert found["body"] == ("fdef", "int(int)")
    assert files == {"include/t.H", "src/u.CPP"}


def test_census_table_has_the_clash_shape(tmp_path: Path) -> None:
    world = tree(tmp_path)
    census = typeunify.collect(world.found(), world.read)
    assert set(typecensus.clashes(census.per)) == {"g", "h"}
    rows = typecensus.table(census.per, typecensus.index({"src/a.CPP": "g g", "include/h.H": "g h"})).splitlines()
    assert rows[0] == typecensus.HEADER.rstrip("\n")
    cols = rows[1].split("\t")
    assert cols[:2] == ["g", "2"] and "long => src/a.CPP:1 ext [de,us]" in cols[2]
    assert cols[3] == "2: include/h.H src/a.CPP" and cols[4] == "open"


# ---------------------------------------------------------------- closure, plan, batching
def test_closure_is_the_files_a_unit_reads(tmp_path: Path) -> None:
    world = tree(tmp_path)
    census = typeunify.collect(world.found(), world.read)
    assert census.closure[("us", "src/a.CPP")] == {"include/h.H", "src/a.CPP"}


def test_plan_orders_candidates_and_rewrites_the_other_declarations(tmp_path: Path) -> None:
    world = tree(tmp_path)
    census = typeunify.collect(world.found(), world.read)
    src = typeplan.Sources(lambda p: (tmp_path / p).read_text())
    sites = typeplan.sites_of(census.per, "g")
    assert typeplan.candidates(sites) == ["long", "int"]  # equal votes: the longer spelling first
    first = typeplan.plan("g", sites, 0, src, census.norm)
    second = typeplan.plan("g", sites, 1, src, census.norm)
    assert isinstance(first, typeplan.Plan) and isinstance(second, typeplan.Plan)
    assert typeplan.render("include/h.H", [first], src) == "extern long g;\nextern int h;\n"
    assert typeplan.render("src/a.CPP", [second], src).startswith("extern int g;\n")
    assert typeplan.plan("g", sites, 2, src, census.norm) == typeplan.EXHAUSTED


def test_a_definition_decides_alone() -> None:
    sites = {"a.C:1": ("vdef", "int", 1), "b.H:2": ("ext", "short", 1), "c.H:3": ("ext", "short", 1)}
    assert typeplan.candidates(sites) == ["int"]
    voted = {"a.H:1": ("ext", "void*", 1), "b.H:1": ("ext", "int*", 1), "c.H:1": ("ext", "int*", 1)}
    assert typeplan.candidates(voted) == ["int*", "void*"]


def test_multi_declarator_statement_keeps_the_other_name(tmp_path: Path) -> None:
    (tmp_path / "include").mkdir()
    (tmp_path / "include/m.H").write_text("extern short a, b;\n")
    src = typeplan.Sources(lambda p: (tmp_path / p).read_text())
    mine = typeplan.Plan("a", "int", "", "int a", (("include/m.H", 0, 18),), frozenset({"include/m.H"}))
    assert typeplan.render("include/m.H", [mine], src) == "extern int a;\nextern short b;\n"


def test_batches_group_disjoint_names() -> None:
    units = {
        "a": frozenset({("us", "x")}),
        "b": frozenset({("us", "y")}),
        "c": frozenset({("us", "x"), ("us", "z")}),
        "none": frozenset(),
    }
    assert typeunify.batches(units) == [["b", "c", "none"], ["a"]]  # c is the widest, a meets it


# ---------------------------------------------------------------- pick, rotation, refusal
def test_pick_keeps_the_most_declared_type_and_batches_disjoint_names(tmp_path: Path) -> None:
    world = tree(tmp_path)
    # g is used by both units, h by b alone: the two share b, so two batches
    result = typeunify.run(tmp_path, world.seams)
    assert result["unified"]["g"]["type"] == "long" and result["unified"]["h"]["type"] == "short"
    assert result["unresolved"] == {} and result["batches"] == 2
    assert (tmp_path / "include/h.H").read_text() == "extern long g;\nextern short h;\n"
    assert len(world.calls) == 1 + 2 + 1  # baseline, one build per batch, the verification of every unit touched


def test_disjoint_names_share_one_build(tmp_path: Path) -> None:
    world = World(
        tmp_path,
        {
            "src/a.CPP": "extern long g;\nextern short g;\nint f(void) { return g; }\n",
            "src/b.CPP": "extern long h;\nextern short h;\nint k(void) { return h; }\n",
        },
        ["src/a.CPP", "src/b.CPP"],
    )
    result = typeunify.run(tmp_path, world.seams)
    assert result["batches"] == 1 and sorted(result["unified"]) == ["g", "h"]


def test_a_failing_type_rotates_to_the_next(tmp_path: Path) -> None:
    world = tree(tmp_path, fail=["extern long g;\nextern"])  # a long g in include/h.H is the fault
    result = typeunify.run(tmp_path, world.seams, ["g"])
    assert result["unified"]["g"]["type"] == "int"
    assert (tmp_path / "include/h.H").read_text().startswith("extern int g;")
    assert (tmp_path / "src/a.CPP").read_text().startswith("extern int g;")  # every declaration took the pick


def test_a_name_no_type_satisfies_is_reported_per_type_and_left_alone(tmp_path: Path) -> None:
    world = tree(tmp_path, fail=["extern long g;", "extern int g;"])
    before = {p: (tmp_path / p).read_text() for p in ("include/h.H", "src/a.CPP")}
    # the baseline must build, so the failing texts are armed after it
    armed = world.fail
    world.fail = []
    original = world.make

    def late(targets: Sequence[str]) -> tuple[int, str]:
        world.fail = armed if 1 <= len(world.calls) <= 2 else []  # the two attempts, not the baseline or the proof
        return original(targets)

    seams = Seams(world.found, world.read, late)
    result = typeunify.run(tmp_path, seams, ["g"])
    assert set(result["unresolved"]["g"]) == {"long", "int"}
    assert all(
        w.endswith("does not build") and w.startswith(("us ", "de ")) for w in result["unresolved"]["g"].values()
    )
    assert {p: (tmp_path / p).read_text() for p in before} == before


@pytest.mark.parametrize(
    ("code", "log", "makes", "why"),
    [
        (0, "", True, "log is empty"),
        (2, "make: *** No rule\n", True, "named no unit"),
        (0, "make\n", False, "neither built"),
    ],
)
def test_a_build_that_did_not_run_is_refused(tmp_path: Path, code: int, log: str, makes: bool, why: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.CPP").write_text("int f;\n")

    def make(targets: Sequence[str]) -> tuple[int, str]:
        for t in targets if makes else ():
            (tmp_path / t).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / t).write_bytes(b"o")
        return code, log

    seams = Seams(lambda: {"us": ["src/a.CPP"]}, lambda items: [], make)
    with pytest.raises(Refusal) as refused:
        typeunify.run(tmp_path, seams)
    assert refused.value.findings[0].key == "unify.build"
    assert why in refused.value.findings[0].reason


def test_a_tree_that_does_not_build_has_no_baseline(tmp_path: Path) -> None:
    world = tree(tmp_path, fail=["extern short h;"])
    with pytest.raises(Refusal) as refused:
        typeunify.run(tmp_path, world.seams)
    assert refused.value.findings[0].key == "unify.baseline"


def test_no_units_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Refusal) as refused:
        typeunify.run(tmp_path, Seams(lambda: {"us": []}, lambda items: [], lambda targets: (0, "x")))
    assert refused.value.findings[0].key == "unify.units"


# ---------------------------------------------------------------- the seams of the speedup
def test_words_are_what_a_unit_defines_or_uses_never_what_it_only_declares() -> None:
    text = (
        '# 1 "include/h.H"\ntypedef int T;\nextern int only_declared;\nint proto(int);\nextern int used;\n'
        "int (*pointer)(int);\nint defined = 3;\nint holder[] = { initialised };\n"
        "int body(void) { return used + proto(1); }\n"
    )
    words = typeunify.facts(text, "src/u.CPP", LIKE)[3]
    assert {"used", "proto", "pointer", "defined", "holder", "initialised", "body"} <= words
    assert "only_declared" not in words


def test_a_name_touches_the_units_that_define_or_use_it(tmp_path: Path) -> None:
    world = tree(tmp_path)
    census = typeunify.collect(world.found(), world.read)
    src = typeplan.Sources(lambda p: (tmp_path / p).read_text())
    plans = {n: typeplan.plan(n, typeplan.sites_of(census.per, n), 0, src, census.norm) for n in ("g", "h")}
    g, h = plans["g"], plans["h"]
    assert isinstance(g, typeplan.Plan) and isinstance(h, typeplan.Plan)
    assert typeunify.reach(census, h) >= {("us", "src/a.CPP"), ("us", "src/b.CPP")}  # the header is in both
    assert typeunify.touched(census, g) == {(v, u) for v in ("us", "de") for u in ("src/a.CPP", "src/b.CPP")}
    assert typeunify.touched(census, h) == {("us", "src/b.CPP"), ("de", "src/b.CPP")}  # a only declares h


def test_only_the_touched_objects_are_made_and_compared(tmp_path: Path) -> None:
    world = tree(tmp_path)
    result = typeunify.run(tmp_path, world.seams, ["h"])
    assert result["unified"]["h"]["units"] == 2
    built = [t for call in world.calls[1:-1] for t in call]
    assert built == ["build/de/src/b.CPP.o", "build/us/src/b.CPP.o"]  # a does not use h


def test_a_unit_the_finer_reading_missed_refuses_the_run_as_drift(tmp_path: Path) -> None:
    world = tree(tmp_path)
    base = world.make

    def late(targets: Sequence[str]) -> tuple[int, str]:
        code, log = base(targets)
        broken = [t for t in targets if "src/a.CPP" in t and len(world.calls) > 1]  # a only declares h: not touched
        for t in broken:
            (tmp_path / t).unlink()
            log += f"make: *** [Makefile:1: {t}] Error 1\n"
        return (2 if broken else code), log

    with pytest.raises(Refusal) as err:
        typeunify.run(tmp_path, Seams(world.found, world.read, late), ["h"])
    assert err.value.findings[0].key == "unify.drift"


def test_differing_names_the_missing_and_the_changed_objects() -> None:
    base = {("us", "a"): "1", ("us", "b"): "2", ("us", "c"): "3"}
    made = {("us", "a"): "1", ("us", "b"): "9", ("us", "c"): None}
    assert typeunify.differing(base, made, base) == [("us", "b"), ("us", "c")]
    assert typeunify.differing([("us", "a")], made, base) == []


def test_nothing_to_rebuild_runs_no_make(tmp_path: Path) -> None:
    world = tree(tmp_path)
    assert typeunify.rebuild(tmp_path, world.seams, []) == {} and world.calls == []


@pytest.mark.parametrize(("slots", "jobs"), [(16, 16), (1, 1), (0, 1)])
def test_a_build_step_asks_for_every_slot_of_the_host(slots: int, jobs: int) -> None:
    host = Host.__new__(Host)
    object.__setattr__(host, "slots", slots)
    assert typeunify.jobs(host) == jobs


def test_a_statement_every_unit_repeats_is_parsed_once_per_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = tree(tmp_path)
    parsed: list[str] = []
    original = Norm.statement

    def count(self: Norm, text: str, hint: str) -> list:  # type: ignore[type-arg]
        parsed.append(text)
        return original(self, text, hint)

    monkeypatch.setattr(Norm, "statement", count)
    census = typeunify.collect(world.found(), world.read)
    header = [t for t in parsed if "extern int g" in t]
    assert len(header) == 2  # the header's statement once per version, not once per unit
    assert set(typecensus.clashes(census.per)) == {"g", "h"}
