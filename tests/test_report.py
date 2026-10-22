"""Progress from the source tree: the credit formula, the README block and the work counts, on mocked inputs."""

import json
from types import SimpleNamespace

import pytest

from unbake import config as configuration
from unbake import report, rescore
from unbake.contracts import Finding, Refusal, sha256

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
    assert tally["code_fuzzy"] == 21 and tally["data_fuzzy"] == 0  # a state (a NON_MATCHING body), never the score
    assert tally["functions_fuzzy"] == 1
    assert tally["fuzzy_percent"] == report._pct(20 + 21, 79)


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
    monkeypatch.setattr(report, "publish_hold", lambda config, tree: None)
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
    monkeypatch.setattr(report, "publish_hold", lambda config, tree: None)
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
    monkeypatch.setattr(
        rescore, "_measure", lambda config, path, names, version: {n: measured.append(n) or 0.5 for n in names}
    )
    monkeypatch.setattr(rescore.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 1  # cold: one measure
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 1  # warm: none
    source.write_text("#ifdef NON_MATCHING\nint f(void) { return 2; }\n#endif\n")
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 2  # changed text: one
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.5}} and len(measured) == 2


def test_readme_with_blob_accounting_measures_against_the_rom_and_drops_the_partition_rows(tmp_path):
    config, current = _readme_case(tmp_path)
    plain = report.readme("<!-- progress -->\n<!-- /progress -->", config, current)
    held = {
        "us": {"rom_size": 8_388_608, "platform_total": 4096, "blobs_total": 1000, "blobs_decoded": 400},
        "eu": {"rom_size": 8_388_608, "platform_total": 4096, "blobs_total": 0, "blobs_decoded": 0},
    }
    for v, row in held.items():
        row["blobs_identified"] = row["blobs_total"] - row["blobs_decoded"]
        row["blobs_fill"] = 0
        row["blobs_no_row"] = 0
        row["unknown_total"] = row["rom_size"] - current["versions"][v]["code_total"] - 4096 - row["blobs_total"]
        current["versions"][v].update(row)
    out = report.readme("<!-- progress -->\n<!-- /progress -->", config, current)
    for row in ("code      [", "data      [", "functions ["):  # values of the old rows are the same
        assert row.strip() in plain and row.strip() in out
    assert "blobs     [░░░░░░░░░░░░░░░░░░░░]   0.00% (~0.01%)  400 of 7,295,964</code>" in out
    for gone in ("platform  [", "unknown   [", "rom       ", "data = C globals"):
        assert gone not in out
    # the top bars are measured against the exact ROM size: matched code+data+blobs plus platform, reaching fuzzy
    assert "us   [" in out and " of 8,388,608 bytes</code>" in out and " of 16,777,216 bytes</code>" in out
    sure = sum(current["versions"]["us"][k] for k in ("code_matched", "data_matched")) + 400 + 4096
    assert f"{sure:,} of 8,388,608 bytes" in out
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


def _extent_cache(monkeypatch):
    held = {}
    monkeypatch.setattr(report.store, "put", lambda config, kind, key, value, named=False: held.__setitem__((kind, key), value))
    monkeypatch.setattr(report.store, "get", lambda config, kind, key, named=False: held.get((kind, key)))
    return held


def test_c2_extents_come_from_the_built_objects_symbols(tmp_path, monkeypatch):
    from unbake import versions
    _extent_cache(monkeypatch)
    module = _module("src/a.c", ["sized", "bare", "last", "gone", "outside"], ["gone"])
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("int sized[8];")
    (tmp_path / "src/b.c").write_text("int x;")
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


def test_extents_follow_the_module_text_to_a_new_path_without_an_object(tmp_path, monkeypatch):
    from unbake import versions
    _extent_cache(monkeypatch)
    (tmp_path / "src/DATA").mkdir(parents=True)
    (tmp_path / "src/DATA/a.c").write_text("int tab[8];")
    (tmp_path / "build/a/src/DATA").mkdir(parents=True)
    (tmp_path / "build/a/src/DATA/a.c.o").write_bytes(b"")
    monkeypatch.setattr(versions, "defined", lambda obj: ({"tab": (".data", 0, 0x20)}, {".data": 0x20}))
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    assert report._extents(config, _tree([_module("src/DATA/a.c", ["tab"])], "old")) == {"a": {"tab": 0x20}}
    (tmp_path / "src/game/DATA").mkdir(parents=True)  # the layout move: same text, new path, no object built there
    (tmp_path / "src/game/DATA/a.c").write_text("int tab[8];")
    assert report._extents(config, _tree([_module("src/game/DATA/a.c", ["tab"])], "new")) == {"a": {"tab": 0x20}}
    (tmp_path / "src/game/DATA/a.c").write_text("int tab[9];")  # other text: never guessed
    assert report._extents(config, _tree([_module("src/game/DATA/a.c", ["tab"])], "new2")) == {}


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


HEADER = (
    bytes.fromhex("80371240 0000000F 80000400 0000144B 12345678 9ABCDEF0 00000000 00000000")
    + b"Example Game        "
    + bytes(7)
    + b"NXXE\x01"
)


def test_header_fields_parse_the_cartridge_header_only():
    from unbake import progress

    assert progress.header(HEADER + b"boot code is never read") == {
        "pi": {"release": 3, "page_size": 7, "pulse_width": 0x12, "latency": 0x40},
        "clock_rate": 15,
        "entry_point": 0x80000400,
        "release": "2.0K",
        "crc1": 0x12345678,
        "crc2": 0x9ABCDEF0,
        "name": "Example Game",
        "game_id": "NXXE",
        "country": "North America",
        "version": 1,
    }
    with pytest.raises(ValueError):
        progress.header(HEADER[:0x3F])


@pytest.fixture
def published_case(tmp_path, monkeypatch):
    from unbake import progress

    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("// source\nvoid exact(void) {}\nvoid fuzzy(void) {}\n")
    config = _config(tmp_path, meta={"a": {"cartridge_id": "NUS-NXXE-0"}})
    (tmp_path / "roms").mkdir()
    (tmp_path / "roms/baserom.a.z64").write_bytes(HEADER + bytes(0x1000 - len(HEADER)))
    tree = _tree([_module("src/a.c", ["exact", "alias"], ["fuzzy", "asm"], ["fuzzy"])], "publish")
    facts = _facts(exact=(4096, 16, "func"), alias=(4100, 0, "func"), fuzzy=(4112, 8, "func"), asm=(4120, 8, "func"))
    rows = _credits(tree, facts, scored={"fuzzy": {"a": 1.0}})
    monkeypatch.setattr(report, "_capture", lambda config: tree)
    monkeypatch.setattr(report, "publish_hold", lambda config, tree: None)
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    monkeypatch.setattr(
        report,
        "accounts",
        lambda config, tree: {
            "a": {
                "rom_size": 8192,
                "platform_total": 4096,
                "blobs_total": 64,
                "blobs_decoded": 32,
                "blobs_identified": 32,
                "blobs_fill": 0,
                "unknown_total": 4000,
                "blobs": {"graphics": {"decoded": 1, "identified": 1, "observed": 1}},
            }
        },
    )
    monkeypatch.setattr(report, "layers", lambda config, tree, version: {"platform": [(0, 4096)], "data": []})
    monkeypatch.setattr(
        progress.blobs,
        "load",
        lambda config: {
            "schema": 1,
            "blob": {
                "tex": {"state": "decoded", "type": "graphics", "rom": {"a": [[5000, 5032]]}},
                "other": {"state": "identified", "type": "graphics", "rom": {"a": [[5032, 5064]]}},
                "observed": {"state": "observed", "type": "graphics", "rom": {"a": [[6000, 6032]]}},
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
    assert cart["functions"] == {"matched": 1, "total": 3}
    assert cart["code"] == {"matched": 16, "total": 32}
    assert cart["blobs"] == {"decoded": 32, "total": 64}
    assert cart["header"]["bytes"] == 4096
    assert not {"platform", "unknown"} & set(cart) and "by_type" not in cart["blobs"]
    published_map = json.loads((config.project.root / "docs/map-a.json").read_bytes())
    configuration.validate("progress.map", published_map, "docs/map-a.json")
    assert published_map["schema"] == 1 and published_map["version"] == "a"
    assert published_map["fns"] == [
        ["exact", 4096, 16, 100.0, "src/a.c", 2],
        ["fuzzy", 4112, 8, 100.0, "src/a.c", 3],
        ["asm", 4120, 8, 0.0, None, None],
    ]
    assert published_map["spans"] == {"platform": [[0, 4096]], "data": [], "fill": []}
    assert [row[0] for row in published_map["blobs"]] == ["tex", "other", "observed"]
    assert "map" not in document and "maps" not in document
    assert sorted(p.name for p in (config.project.root / "docs").iterdir()) == ["map-a.json"]
    assert cart["header"]["game_id"] == "NXXE" and cart["header"]["crc1"] == 0x12345678
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


def test_blobs_bar_leaves_the_rest_empty_and_excludes_code_data_platform():
    held = {
        "rom_size": 2000,
        "code_total": 600,
        "data_total": 200,
        "platform_total": 200,
        "blobs_decoded": 200,
        "blobs_total": 500,
        "unknown_total": 500,
    }
    assert report._blobs_line("blobs", held, 20) == (
        "blobs [" + "█" * 4 + "▒" * 6 + "░" * 10 + "]  20.00% (~50.00%)  200 of 1,000"
    )
    assert report._blobs_line("blobs", {"blobs_decoded": 0, "blobs_total": 0, "unknown_total": 1000}, 20) == (
        "blobs [" + "░" * 20 + "]   0.00%  0 of 1,000"
    )


def test_dependencies_are_short_names_linked_to_their_repositories(tmp_path, monkeypatch):
    config, current = _readme_case(tmp_path)
    monkeypatch.setattr(config.project, "toolchain", "cc", raising=False)
    row = {
        "title": "GCC 2.8.1 (SN64)",
        "dependency": {"name": "GCC 2.8.1", "repository": "https://github.com/pmret/gcc-papermario"},
        "downloads": [{"url": "https://x.test/releases/download/master/linux.tar.gz", "sha256": "ab"}],
        "assemble": ["{n64link}"],
    }
    monkeypatch.setattr(configuration, "toolchains", lambda: {"cc": row})
    text = "<!-- progress -->\n<!-- /progress -->\n<!-- dependencies -->\nold\n<!-- /dependencies -->\n"
    out = report.readme(text, config, current)
    block = out.split("<!-- dependencies -->\n")[1].split("<!-- /dependencies -->")[0]
    assert block == (
        "- [GCC 2.8.1](https://github.com/pmret/gcc-papermario)\n- [n64link](https://github.com/AbuN64/mas-n64)\n"
    )
    for bad in ({}, {"name": "GCC 2.8.1", "repository": "https://x.test/releases/download/master/linux.tar.gz"}):
        monkeypatch.setitem(row, "dependency", bad)
        with pytest.raises(Refusal) as refused:
            report.readme(text, config, current)
        assert refused.value.findings[0].key == "repo.dependency"


def test_every_fetched_tool_names_its_repository_without_bracket_text():
    rows = [*configuration.toolchains().values(), *configuration.load_resource("repo.toml")["make"]["tools"].values()]
    fetched = [row for row in rows if row.get("downloads") or row.get("download")]
    assert fetched
    for row in fetched:
        name, repository = row["dependency"]["name"], row["dependency"]["repository"]
        assert not set("()[]") & set(name)
        assert repository.startswith("https://") and "/releases/" not in repository
        assert not repository.endswith((".gz", ".zip", ".xz", ".bz2", ".tar"))


@pytest.mark.parametrize("score,broken", [(None, False), (0.0, False), (0.25, False), (1.0, False), (0.5, True)])
def test_next_fuzzy_state_matches_measured_summary_and_preserves_zero_score_c_links(
    published_case, monkeypatch, score, broken
):
    config, tree = published_case
    facts = _facts(exact=(4096, 16, "func"), alias=(4100, 0, "func"), fuzzy=(4112, 8, "func"), asm=(4120, 8, "func"))
    rows = _credits(
        tree, facts, scored={} if score is None else {"fuzzy": {"a": score}}, broken=("src/a.c",) if broken else ()
    )
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    result = report.run(config, {"next": True})
    configuration.validate("result.report", result, "report")
    expected = int(bool(score) and not broken)
    assert sum(row["state"] == "fuzzy" for row in result["next"]) == expected
    assert result["versions"]["a"]["functions_fuzzy"] == 1  # the three-state count: a NON_MATCHING body is fuzzy
    assert {row["name"]: row["state"] for row in result["next"]} == {
        "asm": "unmatched",
        "fuzzy": "fuzzy" if expected else "unmatched",
    }
    document = json.loads((config.project.root / "progress.json").read_bytes())
    assert document["carts"][0]["functions"] == {"matched": 1, "total": 3}
    published_map = json.loads((config.project.root / "docs/map-a.json").read_bytes())
    body = next(row for row in published_map["fns"] if row[0] == "fuzzy")
    assert body[4:] == ["src/a.c", 3]
    assert {row["name"] for row in result["next"]} == {"asm", "fuzzy"}  # no exact C or zero-byte aliases


def test_files_writes_nothing_without_a_check_verdict_for_the_tree(tmp_path):
    config, _ = _readme_case(tmp_path)
    (tmp_path / "README.md").write_text("<!-- progress -->\nold\n<!-- /progress -->\n")
    tree = _tree([_module("src/a.c", ["f"], versions=("us",))], "unchecked")
    assert report.publish_hold(config, tree).key == "sync.unchecked"
    assert report.files(config, tree) == {}


def _hold_case(tmp_path, document):
    config, _ = _readme_case(tmp_path)
    (tmp_path / "README.md").write_text("<!-- progress -->\nold\n<!-- /progress -->\n")
    (tmp_path / "progress.json").write_text("old json")
    tree = _tree([_module("src/a.c", ["f"], versions=("us",))], "held")
    (tmp_path / ".unbake").mkdir()
    if document is not None:
        (tmp_path / ".unbake/check.json").write_text(
            json.dumps({"digest": "held", "exact": [], "failing": [], **document})
        )
    return config, tree


def test_a_failed_build_verdict_holds_published_progress(tmp_path):
    config, tree = _hold_case(tmp_path, {"build_failed": "make exit 2"})
    hold = report.publish_hold(config, tree)
    assert (hold.key, hold.blocking) == ("sync.build_failed", False)
    assert "make exit 2" in hold.reason
    assert report.files(config, tree) == {}


def test_a_completed_build_verdict_lets_progress_move(tmp_path):
    config, tree = _hold_case(tmp_path, {"build_failed": None})
    assert report.publish_hold(config, tree) is None


def test_record_build_marks_and_clears_the_failure(tmp_path):
    report.record_build(tmp_path, "in", [], "make exit 2")
    assert json.loads((tmp_path / ".unbake/check.json").read_bytes())["build_failed"] == "make exit 2"
    report.record_build(tmp_path, "in", ["us"])
    assert json.loads((tmp_path / ".unbake/check.json").read_bytes())["build_failed"] is None


def test_plain_report_keeps_progress_json_under_the_same_hold(published_case, monkeypatch):
    config, _ = published_case
    path = config.project.root / "progress.json"
    path.write_text("old json")
    finding = Finding("sync.build_failed", "failed", blocking=False)
    monkeypatch.setattr(report, "publish_hold", lambda config, tree: finding)
    result = report.run(config, {})
    assert path.read_text() == "old json"
    assert result["notes"][0]["key"] == "sync.build_failed"


def test_platform_bytes_leave_the_credit_rows_and_each_covering_symbol_is_named(monkeypatch):
    rows = (
        report.Credit("micro", report.Symbol("data", "src/d.c", "c", 0x100), None, "s", "us", 0, 0x300, 0, True, 1.0),
        report.Credit(
            "boot_f", report.Symbol("function", "src/b.c", "c", 0x40), None, "s", "us", 0x80, 0, 0x80, True, 1.0
        ),
        report.Credit("game", report.Symbol("data", "src/g.c", "c", 0x1000), None, "s", "us", 0, 0x10, 0, True, 1.0),
    )
    spans = {"us": [(0x0, 0x60), (0x200, 0x2000)]}  # boot block, then the microcode under micro's tail and game
    got = report.off_platform(rows, spans)
    assert [(r.code, r.data, r.kept, r.platform) for r in got] == [
        (0, 0x100, 0, 0x200),
        (0x60, 0, 0x60, 0x20),
        (0, 0, 0, 0x10),
    ]
    monkeypatch.setattr(report, "credits", lambda config, tree: got)
    found = report.platform_overlaps(None, None)
    assert [f.unit for f in found] == ["micro", "boot_f", "game"] and not any(f.blocking for f in found)
    assert "covers 512 identified platform bytes" in found[0].reason


def test_map_carries_content_blobs_and_validates_with_every_type(monkeypatch):
    from unbake import blobs, progress

    allowed = _schema("blobs")
    types = allowed["properties"]["blob"]["additionalProperties"]["properties"]["type"]["enum"]
    document = {
        "blob": {
            f"b{i}": {"type": t, "state": "observed", "rom": {"a": [[i * 16, i * 16 + 8]]}} for i, t in enumerate(types)
        }
    }
    monkeypatch.setattr(blobs, "load", lambda config: document)
    monkeypatch.setattr(report, "layers", lambda config, tree, version: {})
    row = progress._map(None, None, {"versions": {"a": {"rom_size": 4096}}}, "a", {}, ())
    assert {r[4] for r in row["blobs"]} == set(blobs.CONTENT) and set(blobs.CONTENT) <= set(types)
    configuration.validate("progress.map", row, "docs/map-a.json")
    schema = _schema("progress.map")
    assert schema["properties"]["blobs"]["items"]["prefixItems"][4]["enum"] == list(blobs.CONTENT)


def _schema(name):
    from importlib import resources

    return json.loads((resources.files("unbake.resources") / "schemas" / f"{name}.schema.json").read_text())


_TAIL = (
    "    /* 95884 80294C84 03E00008 */  jr         $ra\n    /* 95888 80294C88 27BD0018 */   addiu     $sp, $sp, 0x18\n"
)
_WHOLE = (
    "    /* 1 80000000 27BDFFE8 */  addiu $sp, $sp, -0x18\n"
    "    /* 2 80000004 03E00008 */  jr $ra\n"
    "    /* 3 80000008 27BD0018 */   addiu $sp, $sp, 0x18\n"
)


def test_fragment_is_a_head_without_a_return_or_a_tail_that_pops_an_unpushed_frame():
    assert report._fragment(_TAIL) and not report._fragment(_WHOLE)
    assert report._fragment("    /* 1 80000000 27BDFFE8 */  addiu $sp, $sp, -0x18\n")
    assert not report._fragment("")


def test_states_are_exact_fuzzy_or_none_and_a_fragment_follows_its_parent(project, monkeypatch):
    tree = _tree([_module("src/a.c", ["ex"], ["fz", "nn", "frag", "hw"], ["fz"])], "states")
    facts = _facts(
        ex=(0x1000, 16, "func"),
        fz=(0x1010, 16, "func"),
        nn=(0x1020, 8, "func"),
        frag=(0x1028, 4, "func"),
        hw=(0x1030, 8, "func"),
        tbl=(0x2000, 30, "data"),
    )
    rows = _credits(tree, facts)
    monkeypatch.setattr(report, "_rows", lambda config, tree, exact, scored: rows)
    texts = {"frag": _TAIL, "hw": "/* Handwritten function */\n" + _WHOLE, "nn": _WHOLE, "fz": _WHOLE}
    monkeypatch.setattr(report, "_asm_text", lambda config, row: texts.get(row.name, ""))
    got = report._states(_config(project), tree)
    assert got["exact_from"] == "assumed"
    a = got["versions"]["a"]
    assert a["functions"] == {"exact": 2, "fuzzy": 1, "none": 1, "handwritten": 1}
    assert a["code"]["none"] == 8 + 4 and a["code"]["exact"] == 16 + 8 and a["code"]["fuzzy"] == 16
    assert a["fragments"] == 1 and [f[0] for f in a["none_fns"]] == ["nn"]
    assert a["data"] == {"exact": 0, "none": 30}
