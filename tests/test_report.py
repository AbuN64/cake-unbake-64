"""Progress from the source tree: the credit formula, the README block and the work counts, on mocked inputs."""

import json
from types import SimpleNamespace

import pytest

from unbake import config as configuration
from unbake import report, rescore
from unbake.contracts import Refusal, sha256

pytestmark = pytest.mark.usefixtures("plain_roms")

SEG = "main"


def _module(path, owns, stand_ins=(), fuzzy=(), versions=("a",)):
    return SimpleNamespace(
        path=path,
        segment=SEG,
        owns={v: (*owns, *stand_ins) for v in versions},
        stand_ins={v: tuple(stand_ins) for v in versions},
        fuzzy=tuple(fuzzy),
    )


def _tree(modules, digest):
    owners = {(v, n): m.path for m in modules for v, names in m.owns.items() for n in names}
    return SimpleNamespace(digest=digest, modules=tuple(modules), owner=lambda v, n: owners.get((v, n)))


def _config(root, versions=("a",), **project):
    meta = project.pop("meta", {})
    rows = {v: SimpleNamespace(meta=meta.get(v, {})) for v in versions}
    return SimpleNamespace(
        project=SimpleNamespace(
            root=root,
            version=rows,
            names_from=versions[0],
            readme_map=project.get("readme_map"),
            readme_map_link=project.get("readme_map_link"),
            toolchain="gcc-2.8.1-sn64",
            file={},
        )
    )


def _facts(**rows):  # name -> (rom, size, type) in version a
    return {"a": {n: (0x80000000 + rom, rom, size, kind) for n, (rom, size, kind) in rows.items()}}


def _credits(tree, facts, exact=("a",), scored=None, broken=()):
    return report.build_credits(["a"], tree, facts, frozenset(exact), scored or {}, frozenset(broken))


def _bucket(**values):
    zero = dict.fromkeys(("code", "data", "functions"), 0)
    return {f"{k}_{s}": values.get(f"{k}_{s}", 0) for k in zero for s in ("total", "matched", "fuzzy")}


@pytest.fixture(autouse=True)
def _no_partition(monkeypatch):  # the partition needs ROMs and segments: test_blobs.py covers it
    monkeypatch.setattr(report, "partitions", lambda config, tree: {})


@pytest.fixture
def project(tmp_path, monkeypatch):
    return tmp_path


@pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.95])
def test_exact_fuzzy_and_stand_in_give_the_old_numbers(project, monkeypatch, score):
    tree = _tree([_module("src/a.c", ["exact"], ["fuzzy", "gone"], ["fuzzy"])], f"d-{score}")
    facts = _facts(
        exact=(0x1000, 20, "func"), fuzzy=(0x1014, 21, "func"), gone=(0x1030, 8, "func"), tbl=(0x2000, 30, "data")
    )
    rows = _credits(tree, facts, scored={"fuzzy": {"a": score}})
    by = {r.name: r for r in rows}
    assert (by["exact"].exact, by["exact"].kept, by["exact"].score) == (True, 20, 0.0)
    assert (by["fuzzy"].exact, by["fuzzy"].kept, by["fuzzy"].score) == (False, 0, score)
    assert (by["gone"].exact, by["gone"].score, by["gone"].member.state) == (False, 0.0, "stand-in")
    assert by["tbl"].member.state == "unowned" and by["tbl"].segment == "unknown"
    assert by["fuzzy"].member.state == "fuzzy"
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    tally = report.current(_config(project), tree)["versions"]["a"]
    assert tally["code_total"] == 49 and tally["code_matched"] == 20
    assert tally["data_total"] == 30 and tally["data_matched"] == 0
    assert tally["functions_total"] == 3 and tally["functions_matched"] == 1
    assert tally["code_fuzzy"] == round(score * 21) and tally["data_fuzzy"] == 0
    assert tally["functions_fuzzy"] == int(score > 0)
    assert tally["fuzzy_percent"] == report._pct(20 + 30 * 0 + round(score * 21), 79)


def test_a_name_is_exact_only_in_a_version_that_links_exact(project):
    tree = _tree([_module("src/a.c", ["f"], versions=("a", "b"))], "links")
    facts = {v: {"f": (0x80000000, 0x1000, 16, "func")} for v in ("a", "b")}
    rows = report.build_credits(["a", "b"], tree, facts, frozenset({"a"}), {}, frozenset())
    assert [(r.version, r.exact, r.kept) for r in rows] == [("a", True, 16), ("b", False, 0)]


def test_a_name_with_no_rom_address_gets_no_row(project):
    tree = _tree([_module("src/a.c", ["f"])], "absolute")
    facts = {"a": {"f": (0x80000000, 0x1000, 16, "func"), "far": (0x810C210C, None, 0, "func")}}
    assert [r.name for r in _credits(tree, facts)] == ["f"]


def test_a_module_that_does_not_compile_earns_no_fuzzy_credit(project):
    tree = _tree([_module("src/a.c", [], ["f"], ["f"])], "broken")
    facts = _facts(f=(0x1000, 16, "func"))
    assert _credits(tree, facts, scored={"f": {"a": 0.5}})[0].score == 0.5
    assert _credits(tree, facts, scored={"f": {"a": 0.5}}, broken=["src/a.c"])[0].score == 0.0


def test_a_missing_score_counts_zero(project):
    tree = _tree([_module("src/a.c", [], ["f"], ["f"])], "missing")
    assert _credits(tree, _facts(f=(0x1000, 16, "func")))[0].score == 0.0


def test_exact_versions_and_failing_come_from_the_check_of_this_tree(tmp_path):
    (tmp_path / ".unbake").mkdir()
    check = tmp_path / ".unbake/check.json"
    config, tree = _config(tmp_path), SimpleNamespace(digest="check-1")
    check.write_text(json.dumps({"digest": "check-1", "exact": ["a"], "failing": ["src/x.c"]}))
    assert report.exact_versions(config, tree) == {"a"} and report._broken(config, tree) == {"src/x.c"}
    other = SimpleNamespace(digest="check-2")  # a check made of another tree says nothing
    assert report.exact_versions(config, other) == frozenset() and report._broken(config, other) == frozenset()


def test_record_keeps_the_verdict_of_this_tree_and_forgets_that_of_another(tmp_path):
    config, tree = _config(tmp_path), SimpleNamespace(digest="t1")
    report.record(config, tree, exact=["a", "b"], failing=[])
    report.record(config, tree, commit="abc", counts={"source.x": 1})  # a source check adds to the same document
    document = json.loads((tmp_path / ".unbake/check.json").read_text())
    assert document == {"commit": "abc", "counts": {"source.x": 1}, "digest": "t1", "exact": ["a", "b"], "failing": []}
    assert report.exact_versions(config, tree) == {"a", "b"}
    later = SimpleNamespace(digest="t2")
    report.record(config, later, commit="def")
    assert report.exact_versions(config, later) == frozenset()  # the link verdict of t1 is not t2's
    assert report.exact_versions(config, tree) == frozenset()  # and the file now judges t2 alone


@pytest.mark.parametrize(
    ("score", "exact", "expected"), [(0.0, False, 0.0), (0.456, False, 45.6), (1.0, False, 100.0), (0.2, True, 100.0)]
)
def test_percent(score, exact, expected):
    credit = report.Credit("f", report.Symbol("function", "", "c", 0), None, "", "a", 4, 0, 0, exact, score)
    assert report.percent(credit) == expected


def test_percent_and_bar_floor():
    assert report._pct(255_780, 255_784) == 99.99 and report._pct(1, 0) == 0.0
    line = report._line("data", _bucket(data_total=255_784, data_matched=255_780), 20, "data")
    assert line == "data [" + "█" * 19 + "░]  99.99%  255,780 of 255,784"
    assert "█" * 20 not in line


def test_partial_cell_floors_and_suffix_only_when_different():
    bucket = _bucket(code_total=1000, code_matched=500, code_fuzzy=74)
    assert report._line("code", bucket, 20, "code").startswith(
        "code [" + "█" * 10 + "▒" + "░" * 9 + "]  50.00% (~57.40%)"
    )
    same = report._line("code", _bucket(code_total=1000, code_matched=500), 20, "code")
    assert same == "code [" + "█" * 10 + "░" * 10 + "]  50.00%  500 of 1,000"


def test_functions_row_has_fuzzy_gradient():
    bucket = _bucket(functions_total=10, functions_matched=5, functions_fuzzy=2)
    assert (
        report._line("functions", bucket, 20, "functions")
        == "functions [" + "█" * 10 + "▒" * 4 + "░" * 6 + "]  50.00% (~70.00%)  5 of 10"
    )


def test_tally_percentages():
    bucket = {**_bucket(code_total=80, code_matched=20, code_fuzzy=10, data_total=20, data_fuzzy=5)}
    tally = report._tally(bucket)
    assert (tally["fuzzy_bytes"], tally["code_percent"], tally["data_percent"]) == (15, 25.0, 0.0)
    assert tally["fuzzy_percent"] == 35.0


def _readme_case(root):
    for v, data in (("us", b"rom-us"), ("eu", b"rom-eu")):
        (root / "roms").mkdir(exist_ok=True)
        (root / "roms" / f"baserom.{v}.z64").write_bytes(data)
    meta = {"us": {"cartridge_id": "NUS-NRWE-0", "region": "North America", "description": "First NTSC release."}}
    config = _config(root, ("us", "eu"), meta=meta)
    values = {
        "us": {**_bucket(), "code_total": 1_088_548, "code_matched": 816_136, "code_fuzzy": 18_070}
        | {"functions_total": 4196, "functions_matched": 3769, "functions_fuzzy": 100},
        "eu": {**_bucket(), "code_total": 4_467_256, "code_matched": 3_219_816, "code_fuzzy": 165_330},
    }
    return config, {"versions": values}


def test_readme_block_exact(tmp_path):
    config, current = _readme_case(tmp_path)
    begin, end = "<!-- progress -->", "<!-- /progress -->"
    sha = {v: sha256(f"rom-{v}".encode()) for v in ("us", "eu")}
    expected = (
        f"{begin}\n"
        "<pre><code>all  [██████████████▒░░░░░]  72.64% (~75.94%)  4,035,952 of 5,555,804 bytes</code>"
        "<br><code>us   [██████████████▒░░░░░]  74.97% (~76.63%)  816,136 of 1,088,548 bytes</code>"
        "<br><code>eu   [██████████████▒░░░░░]  72.07% (~75.77%)  3,219,816 of 4,467,256 bytes</code></pre>\n\n"
        f"| us (NUS-NRWE-0, North America). First NTSC release. SHA256 `{sha['us']}` |\n"
        "|---|\n"
        "| <pre><code>code      [██████████████▒░░░░░]  74.97% (~76.63%)  816,136 of 1,088,548</code>"
        "<br><code>data      [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 0</code>"
        "<br><code>functions [█████████████████▒░░]  89.82% (~92.20%)  3,769 of 4,196</code></pre> |\n\n"
        f"| eu. SHA256 `{sha['eu']}` |\n|---|\n"
        "| <pre><code>code      [██████████████▒░░░░░]  72.07% (~75.77%)  3,219,816 of 4,467,256</code>"
        "<br><code>data      [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 0</code>"
        "<br><code>functions [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 0</code></pre> |\n"
        f"{end}"
    )
    assert report.readme(f"{begin}\nold\n{end}", config, current) == expected


@pytest.mark.parametrize(
    ("link", "href"),
    [(None, None), ("https://site.example/attic/tool/", "https://site.example/attic/tool/#nus-nrwe-0")],
)
def test_readme_embeds_the_overview_of_the_readme_map(tmp_path, link, href):
    config, current = _readme_case(tmp_path)
    config.project.readme_map, config.project.readme_map_link = "us", link
    out = report.readme("<!-- progress -->\n<!-- /progress -->", config, current)
    image = '<img src="docs/overview-us.svg" width="720">'
    assert (f'<a href="{href}">{image}</a>' if href else image) in out


def test_readme_outside_markers_untouched(tmp_path):
    config, current = _readme_case(tmp_path)
    settings = configuration.load_resource("repo.toml")["readme"]
    prefix, suffix = "# Title é\r\n  before\t", "\r\nAfter Ω\n\x00"
    text = prefix + settings["begin"] + "old content\r\n" + settings["end"] + suffix
    rendered = report.readme(text, config, current)
    assert rendered.startswith(prefix + settings["begin"] + "\n")
    assert rendered.endswith("\n" + settings["end"] + suffix)
    assert report.readme(rendered, config, current) == rendered


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<!-- progress -->",
        "<!-- /progress -->",
        "<!-- /progress --><!-- progress -->",
        "<!-- progress --><!-- progress --><!-- /progress -->",
        "<!-- progress --><!-- /progress --><!-- /progress -->",
    ],
)
def test_readme_missing_markers_refuses(tmp_path, text):
    config, current = _readme_case(tmp_path)
    with pytest.raises(Refusal) as error:
        report.readme(text, config, current)
    (finding,) = error.value.findings
    assert (finding.key, finding.reason, finding.path) == (
        "repo.readme_markers",
        "README.md needs one begin and one end progress marker",
        "README.md",
    )


def test_files_lists_only_the_readme_and_the_one_image(tmp_path, monkeypatch):
    config, _ = _readme_case(tmp_path)
    config.project.readme_map = "us"
    (tmp_path / "README.md").write_text("<!-- progress -->\nold\n<!-- /progress -->\n")
    tree = _tree([_module("src/a.c", ["f"], versions=("us", "eu"))], "files")
    facts = {v: {"f": (0x80000000, 0x1000, 16, "func")} for v in ("us", "eu")}
    rows = report.build_credits(["us", "eu"], tree, facts, frozenset({"us"}), {}, frozenset())
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    monkeypatch.setattr(report, "exact_versions", lambda config, tree: frozenset({"us"}))
    made = report.files(config, tree)
    assert sorted(made) == ["README.md", "docs/overview-us.svg"]
    assert made["docs/overview-us.svg"].startswith(b"<svg ")
    assert b"progress" in made["README.md"] and b"old" not in made["README.md"]


def test_files_writes_no_image_when_no_readme_map(tmp_path, monkeypatch):
    config, _ = _readme_case(tmp_path)
    (tmp_path / "README.md").write_text("<!-- progress -->\n<!-- /progress -->\n")
    tree = _tree([], "files-none")
    monkeypatch.setattr(report, "credits", lambda config, tree: ())
    monkeypatch.setattr(report, "exact_versions", lambda config, tree: frozenset())
    assert sorted(report.files(config, tree)) == ["README.md"]


def test_scores_from_cache(tmp_path, monkeypatch):
    (tmp_path / "roms").mkdir()
    (tmp_path / "roms/baserom.a.z64").write_bytes(bytes(range(64)))
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/a.c"
    source.write_text("#ifdef NON_MATCHING\nint f(void) { return 1; }\n#endif\n")
    config = _config(tmp_path)
    tree = _tree([_module("src/a.c", [], ["f"], ["f"])], "cache-1")
    facts = _facts(f=(0x10, 16, "func"))
    stored, measured = {}, []
    monkeypatch.setattr(rescore, "_lookup", lambda config, key: stored.get(key))
    monkeypatch.setattr(rescore, "_record", lambda config, key, value: stored.__setitem__(key, value))
    monkeypatch.setattr(rescore, "_measure", lambda config, path, name, version: measured.append(name) or 0.5)
    monkeypatch.setattr(rescore.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 1  # cold: one measure
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 1  # warm: none
    source.write_text("#ifdef NON_MATCHING\nint f(void) { return 2; }\n#endif\n")
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 2  # changed text: one
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 2


def test_readme_with_blob_accounting_keeps_the_old_rows_and_adds_the_partition(tmp_path):
    config, current = _readme_case(tmp_path)
    plain = report.readme("<!-- progress -->\n<!-- /progress -->", config, current)
    held = {
        "us": {"rom_size": 8_388_608, "boot_total": 4096, "blobs_total": 1000, "blobs_decoded": 400},
        "eu": {"rom_size": 8_388_608, "boot_total": 4096, "blobs_total": 0, "blobs_decoded": 0},
    }
    for v, row in held.items():
        row["blobs_identified"] = row["blobs_total"] - row["blobs_decoded"]
        row["unknown_total"] = row["rom_size"] - current["versions"][v]["code_total"] - 4096 - row["blobs_total"]
        current["versions"][v].update(row)
    out = report.readme("<!-- progress -->\n<!-- /progress -->", config, current)
    for row in ("code      [", "data      [", "functions ["):  # values of the old rows are the same
        assert row.strip() in plain and row.strip() in out
    assert "blobs     [████████▒▒▒▒▒▒▒▒▒▒▒▒]  40.00% (~100.00%)  400 of 1,000</code>" in out
    assert "boot      [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]   0.00% (~100.00%)  0 of 4,096</code>" in out
    assert "unknown   [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 7,294,964</code>" in out
    assert "rom       1,088,548 + 0 + 1,000 + 4,096 + 7,294,964 = 8,388,608 bytes</code>" in out
    assert "blobs  [" in out and "= 16,777,216 bytes" in out  # the all block sums both ROMs
    assert configuration.load_resource("repo.toml")["readme"]["legend"] in out
    assert report.readme(out, config, current) == out


def _owned_tree(owners, stand_ins=()):
    module = _module("src/a.c", list(owners), list(stand_ins))
    return _tree([module], "dedupe"), module


def test_c1_a_name_inside_an_earlier_rows_extent_gets_no_bytes():
    tree, _ = _owned_tree(["obj", "inner", "next"])
    facts = _facts(
        obj=(0x2000, 0x100, "data"),
        alias_at_head=(0x2000, 0x100, "data"),  # a second name of the same bytes
        inner=(0x2040, 0x20, "data"),  # starts inside obj
        next=(0x2100, 0x10, "data"),
    )
    by = {r.name: r for r in _credits(tree, facts)}
    assert [by[n].data for n in ("obj", "alias_at_head", "inner", "next")] == [0x100, 0, 0, 0x10]
    assert sum(r.data for r in by.values()) == 0x110  # each ROM byte is in one row


def test_c1_an_owned_name_wins_over_an_unowned_alias_at_one_address():
    tree, _ = _owned_tree(["zz_owned"])
    facts = _facts(aa_alias=(0x2000, 0x10, "func"), zz_owned=(0x2000, 0x10, "func"))
    by = {r.name: r for r in _credits(tree, facts)}
    assert (by["zz_owned"].code, by["aa_alias"].code) == (0x10, 0)


def test_c2_a_label_inside_a_c_objects_compiled_size_is_credited_to_it():
    tree, _ = _owned_tree(["obj"])
    facts = _facts(
        obj=(0x2000, 0x40, "data"), l1=(0x2040, 0x10, "data"), l2=(0x2050, 0x30, "data"), out=(0x2080, 0x10, "data")
    )
    sizes = {"a": {"obj": 0x80}}  # the object compiles to 0x80 bytes: l1 and l2 are inside it, `out` is not
    by = {r.name: r for r in report.build_credits(["a"], tree, facts, frozenset({"a"}), {}, frozenset(), sizes)}
    assert [(by[n].exact, by[n].member.state, by[n].member.group) for n in ("obj", "l1", "l2")] == [
        (True, "c", "src/a.c")
    ] * 3
    assert (by["out"].exact, by["out"].member.state) == (False, "unowned")
    assert sum(r.data for r in by.values() if r.exact) == 0x80


def test_c2_without_a_measured_extent_or_behind_a_stand_in_a_label_stays_unowned():
    tree, _ = _owned_tree(["obj", "far"], stand_ins=["obj"])
    facts = _facts(
        obj=(0x2000, 0x40, "data"), l1=(0x2040, 0x10, "data"), far=(0x2100, 0x10, "data"), l2=(0x2110, 0x10, "data")
    )
    none = {r.name: r for r in _credits(tree, facts)}  # no built object: nothing is measured
    assert none["l1"].member.state == none["l2"].member.state == "unowned" and not none["l2"].exact
    sizes = {"a": {"obj": 0x80, "far": 0x10}}  # a stand-in has no compiled definition, so its extent is not used
    by = {r.name: r for r in report.build_credits(["a"], tree, facts, frozenset({"a"}), {}, frozenset(), sizes)}
    assert (by["l1"].exact, by["l1"].member.state) == (False, "unowned")
    assert by["l2"].member.state == "unowned"  # `far` is 0x10 long: l2 starts after its extent
    code_tree, _ = _owned_tree(["f"])
    unowned = _credits(code_tree, _facts(f=(0x2000, 8, "func"), lab=(0x2008, 4, "data")))
    assert {r.name: r.member.state for r in unowned}["lab"] == "unowned"


def test_c2_extents_come_from_the_built_objects_symbols(tmp_path, monkeypatch):
    from unbake import versions

    module = _module("src/a.c", ["sized", "bare", "last", "gone", "outside"], ["gone"])
    (tmp_path / "build/a/src").mkdir(parents=True)
    (tmp_path / "build/a/src/a.c.o").write_bytes(b"")
    symbols = {  # name -> (section, offset, st_size); the old compiler writes st_size 0
        "sized": (".data", 0, 0x20),
        "bare": (".data", 0x20, 0),
        "last": (".data", 0x60, 0),
        "gone": (".data", 0x40, 0),
        "outside": (".text", 0, 0),
    }
    monkeypatch.setattr(versions, "defined", lambda obj: (symbols, {".data": 0x80, ".text": 0}))
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    found = report._extents(config, _tree([module], "ext"))["a"]
    assert found == {
        "sized": 0x20,
        "bare": 0x20,
        "last": 0x20,
    }  # bare runs to the next symbol (`gone`, a stand-in, counts)
    assert report._extents(config, _tree([_module("src/b.c", ["x"])], "ext2")) == {}  # not built: no extent


def test_zero_byte_aliases_are_not_functions_and_exact_fuzzy_is_not_matched(project, monkeypatch):
    tree = _tree([_module("src/a.c", ["real", "alias"], ["pending"], ["pending"])], "aliases")
    facts = _facts(real=(0x1000, 16, "func"), alias=(0x1004, 0, "func"), pending=(0x1010, 8, "func"))
    rows = _credits(tree, facts, scored={"pending": {"a": 1.0}})
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    tally = report.current(_config(project), tree)["versions"]["a"]
    assert tally["functions_total"] == 2
    assert tally["functions_matched"] == 1
    assert tally["functions_fuzzy"] == 1
    assert tally["code_matched"] == 16 and tally["code_fuzzy"] == 8


@pytest.fixture
def published_case(tmp_path, monkeypatch):
    from unbake import progress

    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("// source\nvoid exact(void) {}\nvoid fuzzy(void) {}\n")
    config = _config(tmp_path, meta={"a": {"cartridge_id": "NUS-NXXE-0"}})
    tree = _tree([_module("src/a.c", ["exact", "alias"], ["fuzzy", "asm"], ["fuzzy"])], "publish")
    facts = _facts(exact=(4096, 16, "func"), alias=(4100, 0, "func"), fuzzy=(4112, 8, "func"), asm=(4120, 8, "func"))
    rows = _credits(tree, facts, scored={"fuzzy": {"a": 1.0}})
    monkeypatch.setattr(report, "_capture", lambda config: tree)
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    monkeypatch.setattr(
        report,
        "accounts",
        lambda config, tree: {
            "a": {
                "rom_size": 8192,
                "boot_total": 4096,
                "blobs_total": 64,
                "blobs_decoded": 32,
                "blobs_identified": 32,
                "unknown_total": 4000,
                "blobs": {"texture": {"decoded": 1, "identified": 1, "observed": 1}},
            }
        },
    )
    monkeypatch.setattr(report, "layers", lambda config, tree, version: {"boot": [(0, 4096)], "data": []})
    monkeypatch.setattr(
        progress.blobs,
        "load",
        lambda config: {
            "schema": 1,
            "blob": {
                "tex": {"state": "decoded", "type": "texture", "rom": {"a": [[5000, 5032]]}},
                "other": {"state": "identified", "type": "texture", "rom": {"a": [[5032, 5064]]}},
                "observed": {"state": "observed", "type": "texture", "rom": {"a": [[6000, 6032]]}},
            },
        },
    )
    return config, tree


def test_report_publishes_schema2_and_next_from_the_same_credit_rows(published_case):
    config, _ = published_case
    result = report.run(config, {"next": True})
    document = json.loads((config.project.root / "progress.json").read_bytes())
    configuration.validate("progress", document, "progress.json")
    configuration.validate("result.report", result, "report")
    assert document["schema"] == 2
    cart = document["carts"][0]
    assert cart["functions"] == {"matched": 1, "fuzzy": 1, "total": 3}
    assert cart["code"] == {"matched": 16, "fuzzy": 8, "total": 32}
    assert (
        sum((cart["code"]["total"], cart["data"]["total"], cart["blobs"]["total"], cart["boot"], cart["unknown"]))
        == cart["rom_size"]
    )
    assert document["map"]["fns"] == [
        ["exact", 4096, 16, 100.0, "src/a.c", 2],
        ["fuzzy", 4112, 8, 100.0, "src/a.c", 3],
        ["asm", 4120, 8, 0.0, None, None],
    ]
    assert document["map"]["spans"] == {"boot": [[0, 4096]], "data": []}
    assert [row[0] for row in document["map"]["blobs"]] == ["tex", "other", "observed"]
    assert [(r["name"], r["size"]) for r in result["next"]] == [("asm", 8), ("fuzzy", 8)]
    before = (config.project.root / "progress.json").read_bytes()
    assert "next" not in report.run(config, {})
    assert (config.project.root / "progress.json").read_bytes() == before


def test_missing_rom_partition_refuses_without_replacing_published_progress(published_case, monkeypatch):
    config, _ = published_case
    path = config.project.root / "progress.json"
    path.write_bytes(b"previous")
    monkeypatch.setattr(report, "accounts", lambda config, tree: {})
    with pytest.raises(Refusal, match=r"report\.rom"):
        report.run(config, {})
    assert path.read_bytes() == b"previous"
