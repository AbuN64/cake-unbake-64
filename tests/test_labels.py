"""The label reconstructor: runs of the older ROM matched to the current build's tables, with every refusal measured."""

import struct
from types import SimpleNamespace

import pytest

from unbake import config, labels
from unbake.contracts import Refusal


def words(*w: int) -> bytes:
    return b"".join(struct.pack(">I", x) for x in w)


def owners() -> labels.Owners:
    symbols = [
        ("f_a", 0x80001000, 0x100, True),
        ("f_b", 0x80001200, 0x100, True),
        ("d_one", 0x80002000, 0, False),
        ("f_c", 0x80101000, 0x100, True),
    ]
    return labels.Owners(symbols, [(0x80001000, 0x80003000), (0x80101000, 0x80102000)])


def table(*w: int, at: int = 0x80002100, name: str = "t") -> labels.Table:
    own = owners()
    return labels.Table("eu", at, tuple(w), tuple(own.of(x) if x else None for x in w), name)


def test_owners_prefer_a_function_extent_and_else_the_data_symbol_before_in_the_same_section():
    own = owners()
    assert own.of(0x80001040) == labels.Owner("f_a", 0x80001000, True)
    assert own.of(0x80002040) == labels.Owner("d_one", 0x80002000, False)
    assert own.of(0x80001180) is None or own.of(0x80001180).name != "f_b"
    assert own.of(0x80005000) is None


def test_word_kinds_tell_an_address_from_an_offset_without_the_segment_bits():
    assert [labels.word_kind(w) for w in (0, 0x80001000, 0x0024E410, 0x10, 0xFFFFFFFF)] == [
        "zero",
        "address",
        "segmentless",
        "other",
        "other",
    ]


def test_a_run_starts_and_ends_on_an_address_and_a_zero_joins_only_between_two():
    runs, odd = labels.runs_of([0, 0x80001000, 0, 0x80001010, 0, 7, 0x80001020], 0x100)
    assert [(r.rom, r.words) for r in runs] == [(0x104, (0x80001000, 0, 0x80001010)), (0x118, (0x80001020,))]
    assert odd == []


def test_a_segmentless_word_breaks_the_run_and_is_listed_unless_asked_to_join():
    got = [0x80001000, 0x0024E410, 0x80001010]
    runs, odd = labels.runs_of(got, 0)
    assert len(runs) == 2 and odd == [{"rom": 4, "word": "0x0024E410"}]
    runs, odd = labels.runs_of(got, 0, segmentless=True)
    assert len(runs) == 1 and runs[0].words[1] == 0x8024E410 and odd == []


def test_scan_reads_tables_of_owned_addresses_and_normalizes_the_segmentless_form():
    data = words(0x80001000, 0, 0x101010, 0x80001204, 0x80009999, 0x80001000)
    got = labels.scan("eu", [(0x80002100, data)], owners())
    assert [(t.at, t.words) for t in got] == [(0x80002100, (0x80001000, 0, 0x80101010, 0x80001204))] and got[
        0
    ].name == "d_one"


def test_a_run_fits_a_table_with_one_delta_and_the_same_zero_pattern():
    tab = table(0x80001000, 0, 0x80001010, 0x80001204)
    run = labels.Run(0x50, tuple(w + 0x400 if w else 0 for w in tab.words))
    fits, near = labels.candidates(run, [tab])
    assert [(t.name, s, d) for t, s, d in fits] == [("t", 0, 0x400)] and near is None
    moved = labels.Run(0x50, (0x80001400, 0x80001410, 0, 0x80001604))
    assert labels.candidates(moved, [tab])[0] == []


def test_labels_that_disagree_on_the_delta_are_refused_with_the_measured_closest_table():
    tab = table(0x80001000, 0x80001010, 0x80001020, 0x80001204)
    run = labels.Run(0, (0x80001400, 0x80001410, 0x80001420, 0x80001999))
    row, got = labels.decide(run, ["eu"], {"eu": [tab]})
    assert got is None
    assert row["closest"] == {
        "table": "t",
        "table_at": "0x80002100",
        "labels": 4,
        "agree": 3,
        "differ": 1,
        "deltas": 2,
        "version": "eu",
    }


def test_a_match_names_each_label_by_its_owner_and_gives_the_function_start_the_same_delta():
    tab = table(0x80001010, 0x80001020, 0x80001204, 0x80002010)
    run = labels.Run(0, tuple(w + 0x400 for w in tab.words))
    row, got = labels.decide(run, ["eu"], {"eu": [tab]})
    assert row == {"version": "eu", "table": "t", "table_at": "0x80002100", "copies": 1, "delta": 0x400}
    assert got == {
        "g_earlier_f_a": 0x80001400,
        "g_earlier_f_a_10": 0x80001410,
        "g_earlier_f_a_20": 0x80001420,
        "g_earlier_f_b_4": 0x80001604,
        "g_earlier_f_b": 0x80001600,
        "g_earlier_d_one_10": 0x80002410,
    }


def test_copies_that_give_the_same_symbols_decide_and_tables_that_give_other_symbols_make_it_ambiguous():
    one = table(0x80001010, 0x80001020, 0x80001030, 0x80001040)
    run = labels.Run(0, tuple(w + 0x400 for w in one.words))
    same = {"eu": [one, table(*one.words, at=0x80002200, name="u")]}
    assert labels.decide(run, ["eu"], same)[0]["copies"] == 2
    other = table(0x80001010, 0x80001020, 0x80001030, 0x80001040)
    other = labels.Table(
        "eu", 0x80002300, other.words, tuple(labels.Owner("f_z", 0x80001000, True) for _ in other.words), "v"
    )
    row, got = labels.decide(run, ["eu"], {"eu": [one, other]})
    assert got is None and row["why"].startswith("ambiguous") and row["tables"] == 2 and row["differ"]


def test_the_runs_own_version_decides_before_the_others():
    a = table(0x80001010, 0x80001020, 0x80001030, 0x80001040, name="own")
    b = table(0x80001010, 0x80001020, 0x80001030, 0x80001040, name="other")
    run = labels.Run(0, tuple(w + 0x400 for w in a.words))
    assert labels.decide(run, ["eu-x", "eu"], {"eu": [b], "eu-x": [a]})[0]["table"] == "own"
    assert labels.decide(run, ["us", "eu"], {"eu": [b]})[0]["table"] == "other"


def test_a_run_that_fits_nothing_is_refused_without_a_closest_row():
    row, got = labels.decide(labels.Run(0, (0x80001000,) * 4), ["eu"], {})
    assert got is None and row == {"why": "no current table has these entries with one delta"}


def test_one_name_with_two_addresses_refuses_both_runs_and_two_names_for_one_address_stay():
    run = labels.Run(0, (0x80001000,))
    matched = [(run, {}, {"a": 1, "x": 5}), (run, {}, {"a": 2}), (run, {}, {"b": 5})]
    keep, bad = labels.resolve(matched)
    assert keep == [2]
    assert [(c["run"], c["why"], c["name"]) for c in bad] == [
        (0, "conflict: one name has two addresses", "a"),
        (1, "conflict: one name has two addresses", "a"),
    ]


def _project(tmp_path, monkeypatch, roms, tables):
    cfg = SimpleNamespace(project=SimpleNamespace(root=tmp_path, name="P", versions=tuple(roms)))
    for v, rom in roms.items():
        path = tmp_path / f"{v}.z64"
        path.write_bytes(rom)
    monkeypatch.setattr(config, "rom", lambda project, v: tmp_path / f"{v}.z64")
    monkeypatch.setattr(labels, "image", lambda root, project, v: tables.get(v))
    return cfg


def _image(*w: int):
    return owners(), [(0x80002100, words(*w))]


def test_the_command_writes_the_symbols_of_a_matched_run_once_with_the_table_as_provenance(tmp_path, monkeypatch):
    cur = (0x80001010, 0x80001020, 0x80001204, 0x80002010)
    rom = b"\0" * 8 + words(*[w + 0x400 for w in cur]) + words(0, 0x80001000, 0x80001010)
    cfg = _project(tmp_path, monkeypatch, {"eu": rom}, {"eu": _image(*cur)})
    params = {"spec": ["eu:8:0x18:0", "eu:0x1c:0x24:0"]}
    got = labels.command(cfg, params)
    assert got["labels"] == 6 and got["symbols"] == {"eu": 6} and got["tables"][0]["table"] == "d_one"
    assert [u["why"] for u in got["unfit"]] == ["fewer than 4 entries identify no table"]
    text = (tmp_path / "versions/eu/earlier_build_symbols.txt").read_text()
    assert "g_earlier_f_a_10 = 0x80001410; // from eu d_one@0x80002100" in text
    # a key:value word in a comment is a splat attribute, so provenance carries none
    assert not [c for c in (ln.partition("//")[2] for ln in text.splitlines()) if ":" in c]
    labels.command(cfg, params)
    assert (tmp_path / "versions/eu/earlier_build_symbols.txt").read_text() == text


def test_check_writes_nothing_and_an_image_less_project_or_missing_spec_refuses(tmp_path, monkeypatch):
    cur = (0x80001010, 0x80001020, 0x80001204, 0x80002010)
    cfg = _project(tmp_path, monkeypatch, {"eu": words(*[w + 0x400 for w in cur])}, {"eu": _image(*cur)})
    assert labels.command(cfg, {"spec": ["eu:0:0x10:0"], "check": True})["labels"] == 6
    assert not (tmp_path / "versions").exists()
    with pytest.raises(Refusal) as none:
        labels.command(cfg, {})
    assert none.value.findings[0].key == "reconstruct-labels.spec"
    monkeypatch.setattr(labels, "image", lambda root, project, v: None)
    with pytest.raises(Refusal) as gone:
        labels.command(cfg, {"spec": ["eu:0:0x10:0"]})
    assert gone.value.findings[0].key == "reconstruct-labels.image"


def test_one_name_may_have_other_addresses_in_other_versions(tmp_path, monkeypatch):
    cur = (0x80001010, 0x80001020, 0x80001204, 0x80002010)
    roms = {"eu": words(*[w + 0x400 for w in cur]), "eu-x": words(*[w + 0x800 for w in cur])}
    cfg = _project(tmp_path, monkeypatch, roms, {"eu": _image(*cur), "eu-x": _image(*cur)})
    got = labels.command(cfg, {"spec": ["eu:0:0x10:0", "eu-x:0:0x10:0"]})
    assert got["symbols"] == {"eu": 6, "eu-x": 6} and got["unfit"] == []
