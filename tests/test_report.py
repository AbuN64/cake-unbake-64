"""W24 report behavior, using immutable records and mocked module boundaries."""

import json
import struct
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import fixture
import pytest

from unbake import adapters, report, rescore, versions
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Group,
    Host,
    LayoutMap,
    Member,
    Placement,
    Project,
    Refusal,
    SdkHit,
    Snapshot,
    UnitSpec,
    Version,
    VersionFiles,
    sha256,
)


@pytest.fixture(autouse=True)
def _compiler_row(monkeypatch):
    row = {"never_emits": ["mfc0", "mtc0", "cache", "sync", "eret"], "fpu_control_with": ["trunc.w.s"]}
    monkeypatch.setattr(adapters, "row", lambda id: row)


def _member(
    name, *, size=16, address=0x80000400, kind="function", group="g", versions=("a", "b"), section=".text", state="asm"
):
    return Member(
        name, kind, state, group, tuple(Placement(v, section, 0x1000, 0x1000 + size, address) for v in versions)
    )


_REAL_SPLIT_SLOT = report.split_slot


def _unit(name, kind="c"):
    return UnitSpec(f"src/{name}.{kind}", kind, "g", (name,), "test", {})


@pytest.fixture
def snapshot_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(report.effort, "stage", lambda name: nullcontext())
    monkeypatch.setattr(
        report.layout, "unit_of", lambda s, m: next((u for u in s.layout.units.values() if m in u.members), None)
    )
    boundary = {k: {"applied": 0, "proposed": 0, "withheld": 0} for k in ("prelude", "split", "merge")}
    monkeypatch.setattr(report.layout, "boundary_plan", Mock(return_value=(None, boundary)))
    monkeypatch.setattr(report, "split_slot", lambda s, m: None)
    monkeypatch.setattr(report.policy, "unclean", lambda s: frozenset())
    monkeypatch.setattr(report.policy, "unclean_rules", lambda s: {})
    monkeypatch.setattr(report.libasm, "recognised", lambda s, m: False)

    def make(members=(), *, units=(), fuzzy=None, groups=None, versions=("a", "b"), overlays=None):
        vf = {v: VersionFiles(f"{v}.z64", "0" * 40, f"{v}.yaml", f"{v}.txt", {}) for v in versions}
        project = Project(
            tmp_path, "fixture", "fixture", "Fixture", versions, versions[0], "test", {}, vf, {}, {}, {}, "project"
        )
        host = Host(
            2,
            2,
            1024,
            1024,
            tmp_path / "cache",
            1024,
            tmp_path / "tools",
            {},
            None,
            2,
            2,
            0.8,
            2,
            ("test", "test@example.invalid"),
            {},
            "host",
        )
        config = Config(project, host, "config")
        groups = (
            groups
            if groups is not None
            else [
                Group("g", "main", tuple(m.name for m in members), "authored", (), False, "src/g.c", "include/main/g.h")
            ]
        )
        lm = LayoutMap(
            {g.name: g for g in groups},
            {m.name: m for m in members},
            {u.path: u for u in units},
            "layout",
            (),
            fuzzy or {},
        )
        vs = {v: Version(v, tmp_path / f"{v}.z64", "a" * 64, vf[v].split, vf[v].symbols, {}, ()) for v in versions}
        marks = configuration.load_resource("repo.toml")["readme"]
        contents = {"README.md": f"{marks['begin']}\nold\n{marks['end']}\n".encode()}
        contents.update(overlays or {})
        return Snapshot(config, "b" * 40, lm, vs, contents, "snapshot")

    return make


def test_credit_is_layout_state(snapshot_factory, monkeypatch):
    members = [_member("credited"), _member("assembly"), _member("ledger_only", state="c")]
    snapshot = snapshot_factory(
        members,
        units=[_unit("credited"), _unit("assembly", "asm")],
        overlays={".unbake/ledger.jsonl": b'{"member":"ledger_only","exact":true}\n'},
    )
    real_read = Snapshot.read
    reads = []

    def read(s, path):
        reads.append(path)
        assert "ledger" not in path
        return real_read(s, path)

    monkeypatch.setattr(Snapshot, "read", read)
    result = report.current(snapshot)
    for version in ("a", "b"):
        assert result["versions"][version] == {
            "code_total": 48,
            "code_matched": 16,
            "data_total": 0,
            "data_matched": 0,
            "functions_total": 3,
            "functions_matched": 1,
            "functions_fuzzy": 0,
            "code_fuzzy": 0,
            "data_fuzzy": 0,
            "fuzzy_bytes": 0,
            "code_percent": 33.33,
            "data_percent": 0.0,
            "fuzzy_percent": 33.33,
        }
        assert result["kinds"]["c"][version]["code_matched"] == 16
        assert result["kinds"]["asm"][version]["code_matched"] == 0
    assert reads == [".unbake/check.json"]


def test_opaque_rom_bytes_stay_out_of_the_denominator(snapshot_factory):
    opaque = _member("blob", size=1000, kind="data", section=".data", state="bin")
    snapshot = snapshot_factory([_member("credited"), opaque], units=[_unit("credited")])
    result = report.current(snapshot)
    for version in ("a", "b"):
        assert result["versions"][version]["data_total"] == 0
        assert result["versions"][version]["fuzzy_percent"] == 100.0
        assert "bin" not in result["kinds"]


def test_unknown_group_counts_as_unknown_segment(snapshot_factory):
    snapshot = snapshot_factory([_member("ungrouped", group="")], groups=[])
    result = report.current(snapshot)
    assert result["segments"]["unknown"] == result["versions"]


@pytest.mark.parametrize(
    "score, code_fuzzy, data_fuzzy, fuzzy_percent",
    [(0.0, 0, 0, 25.0), (0.25, 5, 2, 33.75), (0.5, 10, 4, 42.5), (0.95, 20, 9, 61.25)],
)
def test_fuzzy_bytes_and_percent(snapshot_factory, score, code_fuzzy, data_fuzzy, fuzzy_percent):
    matched = _member("matched", size=20)
    fuzzy = replace(
        _member("fuzzy", size=21),
        placements=(
            Placement("a", ".text", 0x1000, 0x1015, 0x80000400),
            Placement("a", ".rodata", 0x2000, 0x2009, 0x80000500),
            Placement("a", ".bss", 0, 0, 0x80000600, 999),
        ),
    )
    data = _member("data", size=30, kind="data", section=".data", state="data", versions=("a",))
    snapshot = snapshot_factory(
        [matched, fuzzy, data],
        units=[_unit("matched")],
        fuzzy={"fuzzy": {"path": "src/fuzzy/fuzzy.c", "scores": {"a": score}}},
    )
    tally = report.current(snapshot)["versions"]["a"]
    assert tally == {
        "code_total": 41,
        "code_matched": 20,
        "data_total": 39,
        "data_matched": 0,
        "functions_total": 2,
        "functions_matched": 1,
        "functions_fuzzy": int(score > 0),
        "code_fuzzy": code_fuzzy,
        "data_fuzzy": data_fuzzy,
        "fuzzy_bytes": code_fuzzy + data_fuzzy,
        "code_percent": 48.78,
        "data_percent": 0.0,
        "fuzzy_percent": fuzzy_percent,
    }


@pytest.mark.parametrize(
    "document, expected",
    [
        (None, {"stale": True, "action": "unbake check"}),
        (b"not json", {"stale": True, "action": "unbake check"}),
        (b'{"commit":"old","counts":{}}', {"stale": True, "action": "unbake check"}),
        (
            json.dumps({"commit": "b" * 40, "counts": {"debt": 3}}).encode(),
            {"commit": "b" * 40, "counts": {"debt": 3}, "type_debt": {}},
        ),
        (
            json.dumps({"commit": "b" * 40, "counts": {}, "type_debt": {"src/a.c": {"pad-view": 2}}}).encode(),
            {"commit": "b" * 40, "counts": {}, "type_debt": {"src/a.c": {"pad-view": 2}}},
        ),
    ],
)
def test_debt_and_boundary(snapshot_factory, monkeypatch, document, expected):
    snapshot = snapshot_factory(overlays={".unbake/check.json": document})
    counts = {
        "prelude": {"applied": 2, "proposed": 3, "withheld": 1},
        "split": {"applied": 3, "proposed": 3, "withheld": 0},
        "merge": {"applied": 1, "proposed": 3, "withheld": 2},
    }
    monkeypatch.setattr(report.layout, "boundary_plan", Mock(return_value=(None, counts)))
    result = report.current(snapshot)
    assert result["debt"] == expected
    assert result["boundary"] == counts
    assert result["pending_boundary"] == 6
    assert result["commit"] == snapshot.commit
    assert result["versions"]["a"]["code_percent"] == 0.0


def test_items_order_size_then_address(snapshot_factory):
    specs = [
        ("creative", 1, 10),
        ("fuzzy", 1, 10),
        ("tool", 1, 10),
        ("large", 32, 1),
        ("late", 16, 30),
        ("z_tie", 16, 10),
        ("a_tie", 16, 10),
    ]
    members = [_member(n, size=size, address=addr, group="math") for n, size, addr in specs]
    members += [_member("sdk_creative", size=999, group="sdk"), _member("matched", group="math")]
    groups = [
        Group(
            s,
            s,
            tuple(m.name for m in members if m.group == s),
            "authored",
            (),
            False,
            f"src/{s}.c",
            f"include/{s}/{s}.h",
        )
        for s in ("sdk", "math")
    ]
    snapshot = snapshot_factory(
        members,
        groups=groups,
        units=[_unit("matched")],
        fuzzy={"fuzzy": {"path": "src/fuzzy/fuzzy.c", "scores": {"a": 0.7, "b": 0.4}}},
    )
    rows = report.items(snapshot, {})
    assert [r["member"] for r in rows] == [
        "sdk_creative",
        "large",
        "a_tie",
        "z_tie",
        "late",
        "creative",
        "fuzzy",
        "tool",
    ]  # biggest first, sub-16-byte fragments last
    assert {r["segment"] for r in rows} == {"sdk", "math"}
    assert all(r["member"] and r["member"] in snapshot.layout.members for r in rows)
    assert all(set(r) == {"member", "kind", "segment", "size", "address"} for r in rows)
    assert [r["member"] for r in report.items(snapshot, {"segment": "sdk"})] == ["sdk_creative"]


def test_items_candidates_and_holder_fallback(snapshot_factory):
    members = [
        _member("only_b", versions=("b",), address=88),
        _member("data", kind="data", section=".rodata", state="data", size=12),
        _member("bss", kind="data", section=".bss", state="data"),
        _member("no_text", section=".rodata"),
    ]
    rows = report.items(snapshot_factory(members), {})
    assert [r["member"] for r in rows] == ["only_b", "data"]
    assert rows[0]["address"] == 88


def test_items_name_the_existing_source_of_a_withheld_member(snapshot_factory):
    unit = replace(_unit("partial"), withheld=("b",))
    rows = report.items(snapshot_factory([_member("partial")], units=[unit]), {})
    assert [(r["member"], r["source"]) for r in rows] == [("partial", unit.path)]


@pytest.mark.parametrize("kind", ["c", "data", "asm"])
def test_withheld_credit_is_per_version_in_totals_and_objdiff(snapshot_factory, kind):
    datum = kind == "data"
    member = _member("partial", kind="data" if datum else "function", section=".data" if datum else ".text")
    unit = replace(_unit("partial", kind), withheld=("b",))
    snapshot = snapshot_factory([member], units=[unit])
    current = report.current(snapshot)
    prefix = "data" if datum else "code"
    assert current["versions"]["a"][prefix + "_matched"] == (16 if kind != "asm" else 0)
    assert current["versions"]["b"][prefix + "_matched"] == 0
    for version in ["a", "b"]:
        document = report.objdiff(snapshot, current, version)
        expected = version == "a" and kind != "asm"
        assert document["units"][0]["metadata"]["complete"] is expected
        assert int(document["measures"].get("matched_" + prefix, 0)) == (16 if expected else 0)
    assert report.items(snapshot, {})[0]["member"] == "partial"


@pytest.mark.parametrize("modulus", [1, 2, 3, 7, 31])
def test_items_shards_disjoint_and_complete(snapshot_factory, modulus):
    snapshot = snapshot_factory([_member(f"m{i}") for i in range(40)])
    shards = [{r["member"] for r in report.items(snapshot, {"shard": f"{i}/{modulus}"})} for i in range(modulus)]
    all_names = set(snapshot.layout.members)
    assert set.union(*shards) == all_names
    assert sum(map(len, shards)) == len(all_names)
    for i, names in enumerate(shards):
        assert names == {m for m in all_names if int(sha256(m.encode())[:8], 16) % modulus == i}


@pytest.mark.parametrize(
    "shard", ["", "1", "1/0", "0/0", "2/2", "3/2", "-1/2", "0/-2", "1.0/2", "x/2", "0/2/3", " 0/2", "0/2\n", 2, True]
)
def test_items_bad_shard_refuses(snapshot_factory, shard):
    with pytest.raises(Refusal) as error:
        report.items(snapshot_factory(), {"shard": shard})
    assert error.value.findings[0].key == "report.request"


@pytest.mark.parametrize("count", [None, 1, 2, 10, 0, -1, 1.5, "2", True])
def test_items_count(snapshot_factory, monkeypatch, count):
    snapshot = snapshot_factory([_member(f"m{i}") for i in range(3)])
    if count is not None and (type(count) is not int or count < 1):
        with pytest.raises(Refusal) as error:
            report.items(snapshot, {"count": count})
        assert error.value.findings[0].key == "report.request"
        return
    rows = report.items(snapshot, {"count": count})
    assert len(rows) == (3 if count is None else min(count, 3))
    assert [r["member"] for r in rows] == [f"m{i}" for i in range(len(rows))]


@pytest.mark.parametrize("segment", ["not-a-segment", "", 1, True])
def test_items_bad_segment_refuses(snapshot_factory, segment):
    with pytest.raises(Refusal) as error:
        report.items(snapshot_factory(), {"segment": segment})
    assert error.value.findings[0].key == "report.request"


def test_objdiff_validates_and_is_deterministic(snapshot_factory, monkeypatch):
    members = [
        _member("open", address=500, size=30),
        _member("fuzzy", address=300, size=20),
        _member("matched", address=100, size=10, group="sdk"),
        _member("data", address=400, kind="data", section=".data", size=12),
        _member("other_version", versions=("b",)),
    ]
    groups = [
        Group("g", "main", ("open", "fuzzy", "data"), "authored", (), False, "src/g.c", "include/main/g.h"),
        Group("sdk", "boot", ("matched",), "authored", (), False, "src/sdk.c", "include/boot/sdk.h"),
    ]
    snapshot = snapshot_factory(
        members,
        groups=groups,
        units=[_unit("matched"), _unit("data", "data")],
        fuzzy={"fuzzy": {"path": "src/fuzzy/fuzzy.c", "scores": {"a": 0.333333333, "b": 0.5}}},
    )
    validation = Mock(wraps=configuration.validate)
    monkeypatch.setattr(configuration, "validate", validation)
    current = report.current(snapshot)
    first = report.objdiff(snapshot, current, "a")
    second = report.objdiff(snapshot, current, "a")

    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    assert encode(first) == encode(second)
    assert first["version"] == 2  # an unversioned report is migrated by objdiff, doubling every complete_* sum
    assert first["measures"]["complete_code"] == str(
        sum(int(u["measures"].get("complete_code", 0)) for u in first["units"])
    )
    assert [c.args[0] for c in validation.call_args_list].count("objdiff") == 2  # the resource loads validate too
    validation.assert_called_with("objdiff", second, "versions/a/report.json")
    assert [u["name"] for u in first["units"]] == ["matched", "fuzzy", "data", "open"]
    by_name = {u["name"]: u for u in first["units"]}
    assert by_name["matched"]["metadata"] == {
        "complete": True,
        "progress_categories": ["boot"],
        "source_path": "src/matched.c",
    }
    assert by_name["matched"]["measures"] == {
        "total_code": "10",
        "total_units": 1,
        "total_functions": 1,
        "fuzzy_match_percent": 100.0,
        "matched_code": "10",
        "complete_code": "10",
        "matched_code_percent": 100.0,
        "complete_code_percent": 100.0,
        "complete_units": 1,
        "matched_functions": 1,
        "matched_functions_percent": 100.0,
    }
    assert by_name["fuzzy"]["metadata"]["source_path"] == "src/fuzzy/fuzzy.c"
    assert by_name["fuzzy"]["functions"] == [
        {"name": "fuzzy", "size": "20", "address": "300", "fuzzy_match_percent": 33.333332, "metadata": {}}
    ]
    assert by_name["open"]["functions"] == [{"name": "open", "size": "30", "address": "500", "metadata": {}}]
    assert by_name["data"]["functions"] == []
    assert by_name["data"]["measures"]["matched_data"] == "12"
    assert by_name["data"]["sections"] == [
        {"name": ".data", "size": "12", "fuzzy_match_percent": 100.0, "metadata": {}}
    ]
    assert first["measures"]["total_code"] == "60"
    assert first["measures"]["matched_code_percent"] == 16.666666
    # objdiff weights the f32 per-unit value (33.333332), not the unrounded input
    assert first["measures"]["fuzzy_match_percent"] == 27.777777
    assert [c["id"] for c in first["categories"]] == ["boot", "main"]
    assert first["categories"][0]["name"] == "boot"
    assert first["categories"][0]["measures"]["matched_code_percent"] == 100.0


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0.0),
        (100, 100.0),
        (1 / 3, 0.33333334),
        (33.333333333, 33.333332),
        (72.64, 72.64),
        (1.23456789, 1.2345679),
        (1e-30, 1e-30),
    ],
)
def test_f32_shortest(value, expected):
    result = report._f32(value)
    assert result == expected
    assert struct.pack(">f", result) == struct.pack(">f", value)
    digits = len(format(result, ".8e").split("e")[0].rstrip("0").replace(".", "").lstrip("-"))
    for precision in range(1, digits):
        rounded = float(f"{result:.{precision}g}")
        assert struct.pack(">f", rounded) != struct.pack(">f", value)


def _readme_case(snapshot_factory):
    snapshot = snapshot_factory(versions=("us", "eu"))
    files = dict(snapshot.config.project.version_files)
    files["us"] = replace(
        files["us"],
        meta={
            "cartridge_id": "NUS-NRWE-0",
            "region": "North America",
            "description": "First NTSC release (black cartridge).",
        },
    )
    snapshot = replace(
        snapshot, config=replace(snapshot.config, project=replace(snapshot.config.project, version_files=files))
    )
    values = {
        "us": {
            "code_total": 1_088_548,
            "code_matched": 816_136,
            "data_total": 0,
            "data_matched": 0,
            "code_fuzzy": 18_070,
            "data_fuzzy": 0,
            "functions_total": 4196,
            "functions_matched": 3769,
            "functions_fuzzy": 100,
        },
        "eu": {
            "code_total": 4_467_256,
            "code_matched": 3_219_816,
            "data_total": 0,
            "data_matched": 0,
            "code_fuzzy": 165_330,
            "data_fuzzy": 0,
            "functions_total": 0,
            "functions_matched": 0,
            "functions_fuzzy": 0,
        },
    }
    return snapshot, {"versions": values}


def test_readme_block_exact(snapshot_factory):
    snapshot, current = _readme_case(snapshot_factory)
    begin = "<!-- progress -->"
    end = "<!-- /progress -->"
    expected = (
        f"{begin}\n"
        "<pre><code>all  [██████████████▒░░░░░]  72.64% (~75.94%)  4,035,952 of 5,555,804 bytes</code>"
        "<br><code>us   [██████████████▒░░░░░]  74.97% (~76.63%)  816,136 of 1,088,548 bytes</code>"
        "<br><code>eu   [██████████████▒░░░░░]  72.07% (~75.77%)  3,219,816 of 4,467,256 bytes</code></pre>\n\n"
        "| us (NUS-NRWE-0, North America). First NTSC release (black cartridge). SHA256 `" + "a" * 64 + "` |\n"
        "|---|\n"
        "| <pre><code>code      [██████████████▒░░░░░]  74.97% (~76.63%)  816,136 of 1,088,548</code>"
        "<br><code>data      [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 0</code>"
        "<br><code>functions [█████████████████▒░░]  89.82% (~92.20%)  3,769 of 4,196</code></pre> |\n\n"
        "| eu. SHA256 `" + "a" * 64 + "` |\n|---|\n"
        "| <pre><code>code      [██████████████▒░░░░░]  72.07% (~75.77%)  3,219,816 of 4,467,256</code>"
        "<br><code>data      [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 0</code>"
        "<br><code>functions [░░░░░░░░░░░░░░░░░░░░]   0.00%  0 of 0</code></pre> |\n"
        f"{end}"
    )
    assert report.readme(f"{begin}\nold\n{end}", snapshot, current) == expected


def test_readme_outside_markers_untouched(snapshot_factory):
    snapshot, current = _readme_case(snapshot_factory)
    settings = configuration.load_resource("repo.toml")["readme"]
    prefix, suffix = "# Title é\r\n  before\t", "\r\nAfter Ω\n\x00"
    text = prefix + settings["begin"] + "old content\r\n" + settings["end"] + suffix
    rendered = report.readme(text, snapshot, current)
    assert rendered.startswith(prefix + settings["begin"] + "\n")
    assert rendered.endswith("\n" + settings["end"] + suffix)
    assert report.readme(rendered, snapshot, current) == rendered


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
def test_readme_missing_markers_refuses(snapshot_factory, text):
    snapshot, current = _readme_case(snapshot_factory)
    with pytest.raises(Refusal) as error:
        report.readme(text, snapshot, current)
    (finding,) = error.value.findings
    assert (finding.key, finding.reason, finding.path) == (
        "repo.readme_markers",
        "README.md needs one begin and one end progress marker",
        "README.md",
    )


def test_files_canonical_and_complete(snapshot_factory, monkeypatch):
    monkeypatch.setattr(report.pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = snapshot_factory([_member("matched")], units=[_unit("matched")])
    first = report.files(snapshot)
    assert first == report.files(snapshot)
    presented = set(report.present.paths(("a", "b")))
    assert set(first) == {"README.md", "versions/a/report.json", "versions/b/report.json", report.STAMP, *presented}
    current = report.current(snapshot)
    assert {p: first[p] for p in presented} == report.present.files(snapshot, report.progress(snapshot))
    for v in snapshot.config.project.versions:
        content = first[f"versions/{v}/report.json"]
        document = report.objdiff(snapshot, current, v)
        assert json.loads(content) == document
        assert content == report.dump_objdiff(document)
        assert "stamp" not in document  # objdiff refuses a field it does not know
    tool = f"unbake {report.unbake.__version__}"  # the release, never an install digest
    assert json.loads(first[report.STAMP]) == {"tool": tool, "tree": report.stamp(snapshot)["tree"]}
    assert first["README.md"] == report.readme(snapshot.read("README.md").decode(), snapshot, current).encode()


def test_a_report_made_from_another_tree_is_stale(snapshot_factory, monkeypatch):
    monkeypatch.setattr(report.pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    unit = {"src/matched.c": b"int matched;"}
    snapshot = snapshot_factory([_member("matched")], units=[_unit("matched")], overlays=unit)
    written = {p: b for p, b in report.files(snapshot).items() if p.startswith("versions/")}
    same = snapshot_factory([_member("matched")], units=[_unit("matched")], overlays={**unit, **written})
    assert report.stale(same) == []
    edited = {**written, "src/matched.c": b"int matched = 1;"}
    later = snapshot_factory([_member("matched")], units=[_unit("matched")], overlays=edited)
    assert report.stale(later) == ["a", "b"]  # a unit edit no report was made from: every reader refuses it
    broken = snapshot_factory([_member("matched")], units=[_unit("matched")], overlays={**unit, report.STAMP: b"{"})
    assert report.stale(broken) == ["a", "b"]


def test_published_files_never_plan_the_boundaries_they_do_not_show(snapshot_factory, monkeypatch):
    monkeypatch.setattr(report.pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = snapshot_factory([_member("matched")], units=[_unit("matched")])
    expected = report.files(snapshot)
    report.layout.boundary_plan.reset_mock()
    assert report.files(snapshot) == expected
    report.layout.boundary_plan.assert_not_called()  # a whole-program plan, serial, for numbers no file prints
    assert report.progress(snapshot)["versions"] == report.current(snapshot)["versions"]


@pytest.mark.parametrize("next_mode, empty", [(False, False), (False, True), (True, False), (True, True)])
def test_run_results_validate(snapshot_factory, monkeypatch, next_mode, empty):
    snapshot = snapshot_factory([] if empty else [_member("candidate")])
    capture = Mock(return_value=snapshot)
    inbox = Mock(return_value=[fixture.submission(), fixture.submission(member="another")])
    monkeypatch.setattr(report.layout, "capture", capture)
    monkeypatch.setattr(report.land, "inbox", inbox)
    result = report.run(snapshot.config, {"next": next_mode})
    configuration.validate("result.next" if next_mode else "result.report", result, "result")
    capture.assert_called_once_with(snapshot.config)
    if next_mode:
        assert result == {"items": report.items(snapshot, {"next": True})}
        inbox.assert_not_called()
    else:
        assert result["inbox"] == 2
        assert "next" not in result
        inbox.assert_called_once_with(snapshot.config)


def test_items_leave_out_fragments_whose_delay_slot_is_in_the_next_member(snapshot_factory, monkeypatch):
    monkeypatch.setattr(report, "split_slot", lambda snap, name: "next" if name == "cut" else None)
    rows = report.items(snapshot_factory([_member("cut"), _member("whole", address=96)]), {})
    assert [r["member"] for r in rows] == ["whole"]


def test_a_member_that_falls_through_is_a_fragment_and_so_is_its_successor(snapshot_factory, monkeypatch):
    nop, jal, jr = 0, 0x0C000000, 0x03E00008
    rom = {0x1000: [jal, nop], 0x1008: [jr, nop], 0x1010: [jr, nop]}

    def placed(name, start, address):
        return Member(name, "function", "asm", "g", (Placement("a", ".text", start, start + 8, address),))

    snapshot = snapshot_factory(
        [
            placed("call", 0x1000, 0x80000400),
            placed("epilogue", 0x1008, 0x80000408),
            placed("whole", 0x1010, 0x80000410),
        ],
        versions=("a",),
    )
    words = {start + 4 * i: w for start, ws in rom.items() for i, w in enumerate(ws)}
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert report.split_slot(snapshot, "call") == "epilogue"
    assert report.split_slot(snapshot, "epilogue") is None
    assert [r["member"] for r in report.items(snapshot, {})] == ["whole"]


def test_a_one_word_member_that_is_no_jump_runs_into_the_next_and_is_no_target(snapshot_factory, monkeypatch):
    jr, nop, ori = 0x03E00008, 0, 0x34840010  # ori a0, a0, 0x10
    rom = {0x1000: [ori], 0x1004: [jr, nop], 0x100C: [jr, nop]}
    words = {start + 4 * i: w for start, ws in rom.items() for i, w in enumerate(ws)}

    def placed(name, start, size):
        return Member(name, "function", "asm", "g", (Placement("a", ".text", start, start + size, 0x80000000 + start),))

    snapshot = snapshot_factory(
        [placed("piece", 0x1000, 4), placed("rest", 0x1004, 8), placed("empty", 0x100C, 8)], versions=("a",)
    )
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert [report.split_slot(snapshot, n) for n in ("piece", "rest", "empty")] == ["rest", None, None]
    assert [r["member"] for r in report.items(snapshot, {})] == ["empty"]


def test_an_entry_reading_an_undefined_register_is_a_piece_of_a_function(snapshot_factory, monkeypatch):
    jr, nop = 0x03E00008, 0
    prologue = [0x27BDFFE8, 0xAFBF0014, 0xAFB00010, jr, nop]  # addiu sp; sw ra; sw s0; jr ra; nop
    reads_v1 = [0x28620002, 0x10400003, nop, jr, nop]  # slti v0, v1, 2; beqz v0; nop; jr ra; nop
    reads_a0 = [0x28820002, 0x10400003, nop, jr, nop]  # slti v0, a0, 2: an argument
    rom = {0x1000: prologue, 0x1014: reads_v1, 0x1028: reads_a0}
    words = {start + 4 * i: w for start, ws in rom.items() for i, w in enumerate(ws)}

    def placed(name, start):
        return Member(name, "function", "asm", "g", (Placement("a", ".text", start, start + 20, 0x80000000 + start),))

    snapshot = snapshot_factory(
        [placed("normal", 0x1000), placed("middle", 0x1014), placed("argument", 0x1028)], versions=("a",)
    )
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert [report.split_slot(snapshot, n) for n in ("normal", "middle", "argument")] == [
        None,
        "the middle of a function",
        None,
    ]
    assert [r["member"] for r in report.items(snapshot, {})] == ["normal", "argument"]


def test_an_epilogue_only_member_and_a_jump_into_another_member_are_pieces(snapshot_factory, monkeypatch):
    jr, nop = 0x03E00008, 0
    whole = [0x27BDFFE8, 0xAFBF0014, 0x8FBF0014, jr, 0x27BD0018]  # addiu sp,-24; sw ra; lw ra; jr ra; addiu sp,+24
    tail = [jr, 0x27BD0018]  # jr ra; addiu sp,+24: the end of a function opened elsewhere
    leaf = [jr, 0x24020001]  # jr ra; addiu v0, zero, 1: no stack at all
    jump = [0x08000404, nop]  # j 0x1010, which is inside `whole`, not its first word
    call = [0x08000400, nop]  # j 0x1000, the first word of `whole`: a tail call
    rom = {0x1000: whole, 0x1014: tail, 0x101C: leaf, 0x1024: jump, 0x102C: call}
    words = {start + 4 * i: w for start, ws in rom.items() for i, w in enumerate(ws)}

    def placed(name, start, size):
        return Member(name, "function", "asm", "g", (Placement("a", ".text", start, start + size, 0x80000000 + start),))

    snapshot = snapshot_factory(
        [
            placed("whole", 0x1000, 20),
            placed("tail", 0x1014, 8),
            placed("leaf", 0x101C, 8),
            placed("jump", 0x1024, 8),
            placed("call", 0x102C, 8),
        ],
        versions=("a",),
    )
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert [report.split_slot(snapshot, n) for n in ("whole", "tail", "leaf", "jump", "call")] == [
        None,
        "the middle of a function",
        None,
        "the middle of a function",
        None,
    ]
    assert [r["member"] for r in report.items(snapshot, {})] == ["whole", "leaf", "call"]


def test_a_function_whose_words_write_zero_is_data_misclassified_as_code(snapshot_factory, monkeypatch):
    jr, nop = 0x03E00008, 0
    real = [0x27BDFFE8, 0xAFBF0014, nop, jr, nop]  # a nop is not a write to $zero
    load = [0x9400000A, 0x00041080, jr, nop, nop]  # lhu zero, 0xA(zero)
    shift = [0x27BDFFE8, 0x00040080, jr, nop, nop]  # sll zero, a0, 2
    rom = {0x1000: real, 0x1014: load, 0x1028: shift}
    words = {start + 4 * i: w for start, ws in rom.items() for i, w in enumerate(ws)}

    def placed(name, start):
        return Member(name, "function", "asm", "g", (Placement("a", ".text", start, start + 20, 0x80000000 + start),))

    snapshot = snapshot_factory(
        [placed("real", 0x1000), placed("load", 0x1014), placed("shift", 0x1028)], versions=("a",)
    )
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert [report.split_slot(snapshot, n) for n in ("real", "load", "shift")] == [
        None,
        "data misclassified as code",
        "data misclassified as code",
    ]
    assert [r["member"] for r in report.items(snapshot, {})] == ["real"]


def test_reading_zero_as_a_source_keeps_the_function(snapshot_factory, monkeypatch):
    jr, nop = 0x03E00008, 0
    li = 0x24040005
    store = 0xAFA00010  # sw zero, 0x10(sp)
    move = 0x00001025  # or v0, zero, zero
    sltu = 0x0004102B  # sltu v0, zero, a0
    beq = 0x10000001  # b: both operands $zero
    body = [li, store, move, sltu, beq, nop, jr, nop]
    words = {0x1000 + 4 * i: w for i, w in enumerate(body)}
    snapshot = snapshot_factory(
        [
            Member(
                "reader", "function", "asm", "g", (Placement("a", ".text", 0x1000, 0x1000 + 4 * len(body), 0x80001000),)
            )
        ],
        versions=("a",),
    )
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert report.split_slot(snapshot, "reader") is None
    assert [r["member"] for r in report.items(snapshot, {})] == ["reader"]


def test_items_leave_out_handwritten_assembly(snapshot_factory):
    rows = report.items(snapshot_factory([_member("loop", state="hasm"), _member("plain", address=96)]), {})
    assert [r["member"] for r in rows] == ["plain"]


def test_a_function_with_a_coprocessor_or_system_instruction_is_not_offered(snapshot_factory, monkeypatch):
    jr, nop = 0x03E00008, 0
    bodies = {
        "plain": [0x27BDFFE8, jr, nop],
        "status": [0x40026000, 0x34420001, 0x40826000, jr, nop],  # mfc0, or, mtc0
        "flush": [0xBC000000, jr, nop],
        "return": [0x42000018, nop, nop],
        "order": [0x0000000F, jr, nop],
    }
    words, members, start = {}, [], 0x1000
    for name, body in bodies.items():
        words.update({start + 4 * i: w for i, w in enumerate(body)})
        members.append(
            Member(
                name,
                "function",
                "asm",
                "g",
                (Placement("a", ".text", start, start + 4 * len(body), 0x80000000 + start),),
            )
        )
        start += 0x20
    snapshot = snapshot_factory(members, versions=("a",))
    monkeypatch.setattr(
        versions, "rom_bytes", lambda v, lo, hi: b"".join(words[a].to_bytes(4, "big") for a in range(lo, hi, 4))
    )
    monkeypatch.setattr(report, "split_slot", _REAL_SPLIT_SLOT)
    assert [report.split_slot(snapshot, n) for n in bodies] == [None] + ["system code C cannot emit"] * 4
    assert [r["member"] for r in report.items(snapshot, {})] == ["plain"]


def test_a_fuzzy_score_missing_for_a_holding_version_counts_zero(snapshot_factory):
    fuzzy = _member("fuzzy", size=20)
    snapshot = snapshot_factory([fuzzy], fuzzy={"fuzzy": {"path": "src/fuzzy/fuzzy.c", "scores": {"a": 0.5}}})
    versions = report.current(snapshot)["versions"]
    assert versions["a"]["code_fuzzy"] == 10 and versions["b"]["code_fuzzy"] == 0


def _unclean_snapshot(snapshot_factory, monkeypatch, texts):
    members = [_member("dirty"), _member("clean")]
    snapshot = snapshot_factory(
        members, units=[_unit("dirty"), _unit("clean")], overlays={f"src/{n}.c": t.encode() for n, t in texts.items()}
    )
    monkeypatch.setattr(report.policy, "unclean", real_unclean)
    monkeypatch.setattr(report.policy, "unclean_rules", real_unclean_rules)
    monkeypatch.setattr(report.policy.pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    monkeypatch.setattr(report.policy.store, "cached", lambda c, k, key, produce: produce())
    return snapshot


real_unclean = report.policy.unclean
real_unclean_rules = report.policy.unclean_rules


@pytest.mark.parametrize("ids, moved", [(["struct-view-cast"], True), (["raw-offset"], False)])
def test_an_unclean_exact_function_counts_fuzzy_only(snapshot_factory, monkeypatch, ids, moved):
    load = configuration.load_resource
    monkeypatch.setattr(
        configuration, "load_resource", lambda n: {**load(n), "unclean": ids} if n == "rules.toml" else load(n)
    )
    snapshot = _unclean_snapshot(
        snapshot_factory,
        monkeypatch,
        {"dirty": "void f(void *p) { ((func_80001234_S0 *)p)->x = 1; }\n", "clean": "void g(void) {}\n"},
    )
    result = report.current(snapshot)["versions"]["a"]
    assert (result["code_matched"], result["code_fuzzy"], result["functions_matched"]) == (
        (16, 16, 1) if moved else (32, 0, 2)
    )
    assert result["fuzzy_percent"] == 100.0


_SHARED = (
    "typedef struct { int x; } Fake_func_80001234_S0;\n"  # a file-scope fake type that only `reads` names
    "static Fake_func_80001234_S0 *gFake;\n"
    "int reads(void) {\n    return gFake->x;\n}\n"
    "int other(void *p) {\n    return 1;\n}\n"
)
_OWN = "int reads(void *p) {\n    return *(int *)((u8 *)p + 4);\n}\nint other(void) {\n    return 1;\n}\n"


@pytest.mark.parametrize("ids, text", [(["raw-offset"], _OWN), (["fake-type"], _SHARED)])
def test_an_unclean_finding_withholds_only_the_function_that_holds_or_reads_it(
    snapshot_factory, monkeypatch, ids, text
):
    load = configuration.load_resource
    monkeypatch.setattr(
        configuration, "load_resource", lambda n: {**load(n), "unclean": ids} if n == "rules.toml" else load(n)
    )
    unit = UnitSpec("src/both.c", "c", "g", ("reads", "other"), "test", {})
    snapshot = snapshot_factory(
        [_member("reads"), _member("other")], units=[unit], overlays={"src/both.c": text.encode()}
    )
    monkeypatch.setattr(report.policy, "unclean", real_unclean)
    monkeypatch.setattr(report.policy, "unclean_rules", real_unclean_rules)
    monkeypatch.setattr(report.policy.pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    assert report.policy.unclean_rules(snapshot, {"src/both.c"}) == {"reads": (f"source.{ids[0]}",)}
    result = report.current(snapshot)["versions"]["a"]
    assert (result["code_matched"], result["code_fuzzy"], result["functions_matched"]) == (16, 16, 1)
    rows = {
        f["name"]: f for u in report.objdiff(snapshot, report.current(snapshot), "a")["units"] for f in u["functions"]
    }
    assert rows["other"]["fuzzy_match_percent"] == rows["reads"]["fuzzy_match_percent"] == 100.0


def test_a_finding_no_function_names_withholds_every_function_of_the_source():
    text = "int a(void) { return 1; }\nint b(void) { return 2; }\n"
    found = [Finding("source.union-view", "a union: union", path="src/x.c", line=9)]
    assert report.policy.withheld(text, found, ["a", "b"]) == {"a": ("source.union-view",), "b": ("source.union-view",)}


def test_a_declaration_naming_a_withheld_type_carries_it_to_the_functions_that_read_it():
    text = (
        "typedef struct { int x; } Fake_func_80001234_S0;\n"
        "typedef struct { Fake_func_80001234_S0 *p; } Wrap;\n"
        "int reads(Wrap *w) { return w->p->x; }\n"
        "int other(void) { return 2; }\n"
    )
    found = [Finding("source.fake-type", "invented: Fake_func_80001234_S0", path="src/x.c", line=1)]
    assert report.policy.withheld(text, found, ["reads", "other"]) == {"reads": ("source.fake-type",)}


def _bucket(**values):
    zero = dict.fromkeys(("code", "data", "functions"), 0)
    return {f"{k}_{s}": values.get(f"{k}_{s}", 0) for k in zero for s in ("total", "matched", "fuzzy")}


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


@pytest.mark.parametrize("broken", [False, True])
def test_objdiff_preserves_source_and_zero_score_for_unproved_compiled_units(snapshot_factory, monkeypatch, broken):
    unit = replace(_unit("sourced"), withheld=() if broken else ("b",))
    snapshot = snapshot_factory([_member("sourced")], units=[unit])
    monkeypatch.setattr(report, "_broken", lambda s: frozenset({unit.path}) if broken else frozenset())
    document = report.objdiff(snapshot, report.current(snapshot), "b")
    row = document["units"][0]
    assert row["metadata"]["source_path"] == unit.path
    assert row["metadata"]["complete"] is False
    assert row["functions"][0]["fuzzy_match_percent"] == 0.0
    assert int(document["measures"]["matched_code"]) == 0


def test_map_command_writes_the_documented_map_once(snapshot_factory, monkeypatch, tmp_path):
    import jsonschema

    rom = bytearray(0x2000)
    rom[0x1010:0x1018] = b"hello!!\0"
    (tmp_path / "a.z64").write_bytes(bytes(rom))
    (tmp_path / "b.z64").write_bytes(bytes(rom))
    code = _member("matched", address=0x80000400)
    table = _member("table", kind="data", state="rodata", section=".rodata", address=0x80000410)
    table = replace(table, placements=tuple(replace(p, rom_start=0x1010, rom_end=0x1020) for p in table.placements))
    mapped = [Group("g", "main", ("matched", "table"), "authored", (), False, "src/g.c", "include/main/g.h")]
    snapshot = snapshot_factory([code, table], units=[_unit("matched")], groups=mapped)
    snapshot = replace(
        snapshot,
        versions={
            v: replace(r, segments=(("main", 0x1000, 0x2000, 0x80000400),)) for v, r in snapshot.versions.items()
        },
    )
    row = ("matched", frozenset({0x80000410}), frozenset({"other"}), True, {})
    writes = Mock(return_value=True)
    monkeypatch.setattr(report.layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(report.infer, "scan", Mock(return_value={"a": [row], "b": [row]}))
    monkeypatch.setattr(report.infer, "sdk", lambda s: {"matched": SdkHit("osMatched", "matched", "libgultra")})
    monkeypatch.setattr(report.store, "write", writes)
    monkeypatch.setattr(report.headers, "included_by", lambda s: {"include/x.h": {"src/matched.c"}})
    monkeypatch.setattr(report.headers, "declares", lambda s: {"include/x.h": ["matched"]})
    monkeypatch.setattr(report.headers, "sources", lambda s, place="include", suffix=".h": [f"{place}/x{suffix}"])
    result = report.map_command(snapshot.config, {})
    assert result["path"] == ".unbake/map.json" and result["counts"]["members"] == 2 and writes.call_count == 1
    document = json.loads(writes.call_args.args[1])
    schema = configuration._schema("map")
    jsonschema.validate(document, schema)
    member = next(m for m in document["members"] if m["name"] == "matched")
    assert (member["calls"], member["loads"], member["strings"]) == (["other"], ["table"], ["hello!!"])
    assert member["unit"] == "src/matched.c" and member["sdk"] and member["padded"]
    assert (member["sdk_name"], member["sdk_module"], member["sdk_library"]) == ("osMatched", "matched", "libgultra")
    assert document["headers"][0]["included_by"] == ["src/matched.c"]


def test_map_moves_writes_the_restructure_stream_for_library_and_game_modules(snapshot_factory, monkeypatch, tmp_path):
    (tmp_path / "a.z64").write_bytes(bytes(0x2000))
    (tmp_path / "b.z64").write_bytes(bytes(0x2000))
    funcs = [_member("lib", address=0x80000400), _member("game", address=0x80000500)]
    groups = [
        Group("lib_g", "main", ("lib",), "authored", (), False, "src/lib_g.c", "include/main/lib_g.h"),
        Group("game_g", "main", ("game",), "authored", (), False, "src/game_g.c", "include/main/game_g.h"),
    ]
    snapshot = snapshot_factory(funcs, units=[], groups=groups)
    snapshot = replace(
        snapshot,
        versions={
            v: replace(r, segments=(("main", 0x1000, 0x2000, 0x80000400),)) for v, r in snapshot.versions.items()
        },
    )
    writes = Mock(return_value=True)
    monkeypatch.setattr(report.layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(report.infer, "scan", Mock(return_value={}))
    monkeypatch.setattr(report.infer, "sdk", lambda s: {"lib": SdkHit("osLib", "lib", "libgultra")})
    monkeypatch.setattr(report.store, "write", writes)
    monkeypatch.setattr(report.headers, "included_by", lambda s: {})
    monkeypatch.setattr(report.headers, "declares", lambda s: {})
    monkeypatch.setattr(report.headers, "sources", lambda s, place="include", suffix=".h": [])
    result = report.map_command(snapshot.config, {"moves": True})
    assert result["moves"] == {
        "path": ".unbake/moves.jsonl",
        "count": 4,
        "folders": {"src/game/sys_80000500": 1, "src/sdk/libgultra": 1},
    }
    stream = [json.loads(line) for line in writes.call_args.args[1].decode().splitlines()]
    assert stream == [
        {"layout": snapshot.layout.digest},  # the stream names the layout it was made for
        {"op": "source", "subject": "game_g", "target": "src/game/sys_80000500/game_g.c"},
        {"op": "source", "subject": "lib_g", "target": "src/sdk/libgultra/lib_g.c"},
        {"op": "header", "subject": "game_g", "target": "include/game/sys_80000500.h"},
        {"op": "header", "subject": "lib_g", "target": "include/sdk/libgultra.h"},
    ]


def test_map_segments_fold_per_version_spans_and_cover_a_function_only_segment(snapshot_factory, monkeypatch, tmp_path):
    (tmp_path / "a.z64").write_bytes(bytes(0x2000))
    (tmp_path / "b.z64").write_bytes(bytes(0x2000))
    here, there = _member("here", address=0x80000400), _member("there", address=0x80100000)
    snapshot = snapshot_factory([here, there], units=[])
    spans = {
        "a": (("main", 0x1000, 0x2000, 0x80000400), ("ovl", 0x2000, 0x3000, 0x80100000)),
        "b": (("span_b", 0x1100, 0x2100, 0x80000400),),
    }
    snapshot = replace(snapshot, versions={v: replace(r, segments=spans[v]) for v, r in snapshot.versions.items()})
    writes = Mock(return_value=True)
    monkeypatch.setattr(report.layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(report.infer, "scan", Mock(return_value={}))
    monkeypatch.setattr(report.infer, "sdk", lambda s: frozenset())
    monkeypatch.setattr(report.store, "write", writes)
    monkeypatch.setattr(report.headers, "included_by", lambda s: {})
    monkeypatch.setattr(report.headers, "declares", lambda s: {})
    monkeypatch.setattr(report.headers, "sources", lambda s, place="include", suffix=".h": [])
    report.map_command(snapshot.config, {})
    document = json.loads(writes.call_args.args[1])
    by = {s["name"]: s["versions"] for s in document["segments"]}
    assert by == {
        "main": {"a": [0x1000, 0x2000, 0x80000400], "b": [0x1100, 0x2100, 0x80000400]},
        "ovl": {"a": [0x2000, 0x3000, 0x80100000]},
    }
    assert {m["name"]: (m["segment"], m["sdk_name"]) for m in document["members"]} == {
        "here": ("main", ""),
        "there": ("ovl", ""),
    }


class _Measured(list):
    dispatched: list


def _fuzzy_rows(snapshot_factory, monkeypatch, keys):
    """Two fuzzy rows (f has key 'old', g has key 'same'); every candidate now keys to `keys`."""
    rows = {
        "f": {"path": "src/fuzzy/f.c", "scores": {"a": 0.25}, "key": {"a": "old"}},
        "g": {"path": "src/fuzzy/g.c", "scores": {"a": 0.5}, "key": {"a": "same"}},
    }
    snapshot = snapshot_factory([_member("f", size=20), _member("g", size=20)], units=[], fuzzy=rows)
    monkeypatch.setattr(rescore.effort, "stage", lambda name: nullcontext())
    monkeypatch.setattr(rescore.compare, "bind", Mock(side_effect=lambda snap, file, name: (name, snap)))
    monkeypatch.setattr(rescore.compare, "fuzzy_keys", Mock(side_effect=lambda cfg, bound: [keys[u] for _, u in bound]))
    measured = _Measured()
    cache, dispatched = {}, []

    def fake_map(config, name, function, items, key=None):
        out = []
        for item in items:
            if key is None:  # an unkeyed job (the measure of a moved row) always runs
                out.append(function(item))
                continue
            k = key(item)
            if k not in cache:
                dispatched.append(item[1])
                cache[k] = function(item)
            out.append(cache[k])
        return out

    monkeypatch.setattr(rescore.pool, "map", fake_map)
    measured.dispatched = dispatched

    def measure(config, bound, overrides):
        measured.append([u for _, u in bound])
        return [(fixture.proof(f"src/fuzzy/{u}.c", u, "a", False, ("bytes",), 0.75),) for _, u in bound]

    monkeypatch.setattr(rescore.compare, "measure_bound", measure)
    return snapshot, measured


def test_a_drain_rescores_only_the_rows_whose_key_moved_in_one_dispatch(snapshot_factory, monkeypatch):
    snapshot, measured = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "new"}, "g": {"a": "same"}})
    fuzzy = rescore.rows(snapshot)
    assert measured == [["f"]]
    assert fuzzy["f"] == {"path": "src/fuzzy/f.c", "scores": {"a": 0.75}, "key": {"a": "new"}}
    assert fuzzy["g"] is snapshot.layout.fuzzy["g"]


def test_an_unchanged_drain_dispatches_no_key_job_and_measures_nothing(snapshot_factory, monkeypatch):
    snapshot, measured = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "old"}, "g": {"a": "same"}})
    rescore.rows(snapshot)
    assert measured.dispatched == ["f", "g"]
    rescore.rows(snapshot)
    assert measured.dispatched == ["f", "g"] and measured == []


def test_a_tool_install_keys_no_fuzzy_row_again(snapshot_factory, monkeypatch):
    """The first landing after an install keyed all 269 rows (186 s) and measured 268 (77 s): rows are facts of their
    text, reach and member rows, never of the tool code."""
    snapshot, measured = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "old"}, "g": {"a": "same"}})
    rescore.rows(snapshot)
    assert measured.dispatched == ["f", "g"]
    monkeypatch.setattr(rescore, "code_identity", lambda *functions: "another install")
    rescore.effort._memo.clear()
    rescore.rows(snapshot)
    assert measured.dispatched == ["f", "g"] and measured == []


def test_editing_one_candidate_runs_one_key_job_and_measures_one_member(snapshot_factory, monkeypatch, tmp_path):
    snapshot, measured = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "old"}, "g": {"a": "same"}})
    texts = {"src/fuzzy/f.c": b"one", "src/fuzzy/g.c": b"two"}
    monkeypatch.setattr(type(snapshot), "peek", lambda self, path: texts.get(path), raising=False)
    rescore.rows(snapshot)
    measured.dispatched.clear()
    texts["src/fuzzy/f.c"] = b"edited"
    rescore.compare.fuzzy_keys.side_effect = lambda cfg, bound: [
        {"a": "new"} if u == "f" else {"a": "same"} for _, u in bound
    ]
    rescore.rows(snapshot)
    assert measured.dispatched == ["f"] and measured == [["f"]]


def test_report_leaves_the_tree_clean_and_rescores_nothing(snapshot_factory, monkeypatch):
    snapshot, measured = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "new"}, "g": {"a": "same"}})
    monkeypatch.setattr(report.layout, "capture", Mock(return_value=snapshot))
    monkeypatch.setattr(report.land, "inbox", Mock(return_value=[]))
    applied, written = Mock(), Mock()
    monkeypatch.setattr("unbake.journal.apply", applied)
    monkeypatch.setattr(report.store, "write", written)
    report.run(snapshot.config, {"next": False})
    assert measured == [] and applied.call_count == 0 and written.call_count == 0  # layout.toml is land's to write


def test_the_tally_never_rescores(snapshot_factory, monkeypatch):
    snapshot, measured = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "new"}, "g": {"a": "same"}})
    monkeypatch.setattr(rescore.compare, "bind", Mock(side_effect=AssertionError("a reader bound a row")))
    monkeypatch.setattr(rescore.compare, "fuzzy_keys", Mock(side_effect=AssertionError("a reader keyed a row")))
    totals = report.current(snapshot)["versions"]["a"]
    assert measured == [] and totals["code_fuzzy"] == 5 + 10  # the stored scores, 0.25 and 0.5 of 20 bytes


def test_the_objdiff_pool_is_keyed_by_everything_a_version_report_reads(snapshot_factory, monkeypatch):
    seen = []
    monkeypatch.setattr(
        report.pool,
        "map",
        lambda cfg, name, fn, items, key=None: seen.append((name, key, items)) or [b"{}"] * len(items),
    )
    snapshot = snapshot_factory([_member("matched")], units=[_unit("matched")])
    report.files(snapshot)
    ((name, key, items),) = seen
    assert name == "report.objdiff" and key is report._objdiff_key
    one, other = items[0], items[1]
    assert key(one) != key(other) and key(one) == key(one)
    dirty = (*one[:3], {"src/matched.c": ("source.raw-offset",)}, one[4])
    assert key(dirty) != key(one)  # a source that turned unclean changes the report
    assert key((*one[:4], frozenset({"src/matched.c"}))) != key(one)
    ((_, _, _, rules, broken),) = [items[0]]
    assert rules == {} and broken == frozenset()  # read once by files(), shipped to every version's job


_OBJDIFF = {  # every field objdiff-core's report.proto knows: its JSON reader refuses any other
    "report": {"measures", "units", "version", "categories"},
    "unit": {"name", "measures", "sections", "functions", "metadata"},
    "unit.metadata": {"complete", "module_name", "module_id", "source_path", "progress_categories", "auto_generated"},
    "item": {"name", "size", "fuzzy_match_percent", "metadata", "address"},
    "item.metadata": {"demangled_name", "virtual_address"},
    "category": {"id", "name", "measures"},
}


def test_report_json_holds_only_fields_objdiff_reads(snapshot_factory, monkeypatch):
    monkeypatch.setattr(report.pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    monkeypatch.setattr(report.policy, "unclean_rules", lambda s: {"dirty": ("source.raw-offset",)})
    members = [_member("dirty", size=16), _member("matched"), _member("data", kind="data", section=".data")]
    fuzzy = {"data": {"path": "src/fuzzy/data.c", "scores": {"a": 0.5}, "key": {}}}
    snapshot = snapshot_factory(members, units=[_unit("dirty"), _unit("matched")], fuzzy=fuzzy)
    for path, blob in report.files(snapshot).items():
        if not path.endswith("report.json"):
            continue
        document = json.loads(blob)
        assert set(document) <= _OBJDIFF["report"], path
        assert all(set(c) <= _OBJDIFF["category"] for c in document["categories"])
        for unit in document["units"]:
            assert set(unit) <= _OBJDIFF["unit"] and set(unit["metadata"]) <= _OBJDIFF["unit.metadata"], unit
            for item in (*unit["sections"], *unit["functions"]):
                assert set(item) <= _OBJDIFF["item"] and set(item["metadata"]) <= _OBJDIFF["item.metadata"], item


def test_map_lists_a_group_member_no_version_has_without_a_key_error(snapshot_factory, monkeypatch, tmp_path):
    (tmp_path / "a.z64").write_bytes(bytes(0x2000))
    (tmp_path / "b.z64").write_bytes(bytes(0x2000))
    code = _member("matched", address=0x80000400)
    dangling = [Group("g", "main", ("gone", "matched"), "authored", (), False, "src/g.c", "include/main/g.h")]
    snapshot = snapshot_factory([code], units=[_unit("matched")], groups=dangling)
    snapshot = replace(
        snapshot,
        versions={v: replace(r, segments=(("main", 0, 0x2000, 0x80000400),)) for v, r in snapshot.versions.items()},
    )
    writes = Mock(return_value=True)
    monkeypatch.setattr(report.layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(report.infer, "scan", Mock(return_value={}))
    monkeypatch.setattr(report.infer, "sdk", lambda s: frozenset())
    monkeypatch.setattr(report.store, "write", writes)
    monkeypatch.setattr(report.headers, "included_by", lambda s: {})
    monkeypatch.setattr(report.headers, "declares", lambda s: {})
    monkeypatch.setattr(report.headers, "sources", lambda s, place="include", suffix=".h": [])
    report.map_command(snapshot.config, {})
    module = json.loads(writes.call_args.args[1])["modules"][0]
    assert module["segment"] == "main" and "gone" in module["members"]


def test_moved_rows_bind_and_measure_in_one_worker_dispatch_and_the_parent_binds_none(snapshot_factory, monkeypatch):
    snapshot, _ = _fuzzy_rows(snapshot_factory, monkeypatch, {"f": {"a": "new"}, "g": {"a": "new"}})
    inside, parent_binds, dispatches = [], [], []
    real = rescore.compare.bind

    def bind(snap, file, name):
        if not inside:
            parent_binds.append(name)
        return real(snap, file, name)

    def fake_map(config, name, function, items, key=None):
        dispatches.append((name, len(items)))
        inside.append(1)
        try:
            return [function(i) for i in items]
        finally:
            inside.pop()

    monkeypatch.setattr(rescore.compare, "bind", bind)
    monkeypatch.setattr(rescore.pool, "map", fake_map)
    rescore.rows(snapshot)
    assert dispatches == [("rescore.keys", 2), ("rescore.measure", 2)] and parent_binds == []


def test_a_member_keeps_the_exact_bytes_of_its_clean_functions_when_a_helper_inside_is_withheld(
    snapshot_factory, monkeypatch
):
    monkeypatch.setattr(report.policy, "unclean_rules", lambda s: {"helper": ("source.raw-offset",)})
    monkeypatch.setattr(report.policy, "withheld_bytes", lambda s, rules, name, v: 4 if name == "big" else 0)
    snapshot = snapshot_factory([_member("big", size=16), _member("small")], units=[_unit("big"), _unit("small")])
    current = report.current(snapshot)
    result = current["versions"]["a"]
    assert (result["code_matched"], result["code_fuzzy"], result["functions_matched"]) == (28, 4, 1)
    (big,) = [u for u in report.objdiff(snapshot, current, "a")["units"] if u["name"] == "big"]
    assert big["measures"]["matched_code"] == "12" and not big["metadata"]["complete"]
