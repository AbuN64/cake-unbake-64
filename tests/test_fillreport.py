"""The linker fill in the report (README bars, progress.json) and its audit refusals, over mocked seams (the credit
buckets, the cartridge header, the symbol rows and the object plan)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unbake import fillaudit, linkfill, objectplan, progress, report, versions
from unbake.contracts import sha256
from unbake.linkfill import Fill
from unbake.objectplan import Atom

BEGIN, END = "<!-- progress -->", "<!-- /progress -->"


@pytest.fixture(autouse=True)
def _no_rom(monkeypatch):  # the README names each ROM's hash; the bytes are not the subject here
    monkeypatch.setattr(report, "_rom_sha256", lambda config, version: sha256(version.encode()))


def _bucket(**values: int) -> dict[str, int]:
    keys = [f"{k}_{s}" for k in ("code", "data", "functions") for s in ("total", "matched", "fuzzy", "none", "asm")]
    return {k: values.get(k, 0) for k in keys} | {
        "rom_size": values.get("rom_size", 1000),
        "platform_total": 0,
        "blobs_total": 0,
        "blobs_decoded": 0,
        "blobs_identified": 0,
        "blobs_fill": 0,
        "blobs_no_row": 0,
        "linker_fill_total": values.get("linker_fill_total", 0),
        "unknown_total": 0,
    }


def _config(root: Path) -> SimpleNamespace:
    (root / "roms").mkdir(exist_ok=True)
    meta = {"cartridge_id": "NUS-X-0", "region": "Test"}
    for v in ("us", "eu"):
        (root / "roms" / f"baserom.{v}.z64").write_bytes(b"rom-" + v.encode())
    version = {v: SimpleNamespace(meta=meta) for v in ("us", "eu")}
    project = SimpleNamespace(
        root=root, version=version, readme_map=None, readme_map_link=None, readme_map_repo=None, file={}
    )
    return SimpleNamespace(project=project)


def _current() -> dict:
    us = _bucket(code_total=400, code_matched=100, linker_fill_total=24)
    eu = _bucket(code_total=400, code_matched=100, linker_fill_total=0)
    return {"versions": {"us": us, "eu": eu}}


def test_readme_shows_the_linker_fill_bytes_per_version_and_in_total(tmp_path):
    out = report.readme(f"{BEGIN}\nold\n{END}", _config(tmp_path), _current())
    rows = out.split("</code><br><code>")
    assert rows[0].endswith("200 of 2,000 bytes, linker fill 24 bytes")  # the "all" row sums the versions
    assert rows[1].endswith("100 of 1,000 bytes, linker fill 24 bytes")
    assert rows[2].split("</code></pre>")[0].endswith("100 of 1,000 bytes, linker fill 0 bytes")


def test_the_fill_bytes_never_count_toward_the_matched_source(tmp_path):
    base = _current()
    more = _current()
    more["versions"]["us"]["linker_fill_total"] = 500

    def bar(text: str) -> str:
        row = next(r for r in text.split("<br>") if r.startswith("<code>us"))
        return row.split(", linker")[0]

    assert bar(report.readme(f"{BEGIN}\n{END}", _config(tmp_path), base)) == bar(
        report.readme(f"{BEGIN}\n{END}", _config(tmp_path), more)
    )


def test_progress_cart_carries_the_linker_fill_beside_code_not_in_it(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "_header", lambda config, version: {})
    row = {**_bucket(code_total=400, code_matched=100, linker_fill_total=24), "functions_total": 0}
    row |= {"platform_total": 64}
    config = _config(tmp_path)
    config.project.version["us"].meta = {"cartridge_id": "NUS-X-0"}
    (cart,) = progress.document(config, None, {"versions": {"us": row}})["carts"]
    assert cart["linker_fill"] == {"bytes": 24}
    assert cart["code"]["matched"] == 100 and cart["code"]["total"] == 400
    assert cart["blobs"] == {"decoded": 0, "total": 0}


# ---------------------------------------------------------------- audit
def _fill(rom: int, size: int = 4, obj: str = "7", section: str = ".data") -> Fill:
    return Fill(obj, section, rom, 0x80000000 + rom, b"\xab" * size)


def _sym(name: str, rom: int, size: int | None) -> versions.Row:
    return versions.Row(name, 0x80000000 + rom, "data", rom, size, 1)


def _pads(*fills: Fill) -> set[fillaudit.Pad]:
    return {(f.object, f.section, f.address, f.size) for f in fills}


def test_a_fill_in_a_plan_gap_clear_of_every_symbol_passes():
    fill = _fill(0x20)
    assert fillaudit.findings("us", [fill], [_sym("a", 0x10, 4), _sym("b", 0x24, 8)], _pads(fill)) == []


def test_a_fill_that_overlaps_a_symbol_is_refused_with_a_one_line_cause():
    fill = _fill(0x20)
    (found,) = fillaudit.findings("us", [fill], [_sym("D_table", 0x1C, 8)], _pads(fill))
    assert found.key == "fill.overlap" and found.severity == "Refuse"
    assert (
        found.reason
        == "us: fill at 0x20 (4 bytes, object 7) overlaps symbol D_table; the bytes are the symbol's, not fill"
    )
    assert "\n" not in found.reason


def test_a_symbol_of_no_size_reaches_the_next_symbol_but_stops_at_a_fill_row():
    fill = _fill(0x20)
    assert fillaudit.findings("us", [fill], [_sym("a", 0x10, None)], _pads(fill)) == []  # a ends where the fill starts
    (found,) = fillaudit.findings("us", [fill], [_sym("a", 0x22, None), _sym("b", 0x40, 4)], _pads(fill))
    assert found.unit == "a"


def test_a_fill_for_a_gap_the_plan_does_not_have_is_refused_with_a_one_line_cause():
    fill = _fill(0x20)
    other = _pads(_fill(0x20, obj="8"), _fill(0x20, size=8), _fill(0x20, section=".rodata"), _fill(0x30))
    (found,) = fillaudit.findings("us", [fill], [], other)
    assert found.key == "fill.gap" and found.severity == "Refuse"
    assert (
        found.reason
        == "us: fill at 0x20 (4 bytes, object 7) is in .data at 0x80000020, a gap the object plan does not have"
    )


def test_one_record_can_break_both_rules_and_each_cause_is_named():
    keys = [f.key for f in fillaudit.findings("us", [_fill(0x20)], [_sym("x", 0x20, 4)], [])]
    assert keys == ["fill.overlap", "fill.gap"]


def _project(root: Path, fills: list[Fill], symbols: str) -> None:
    (root / "config.toml").write_text("[version.us]\n[version.eu]\n")
    (root / "versions" / "us").mkdir(parents=True)
    (root / "versions" / "us" / "linker_fill.toml").write_text(linkfill.render("us", fills, "generated"))
    (root / "versions" / "us" / "symbol_addrs.txt").write_text(symbols)


def _plan(monkeypatch, *atoms: Atom) -> None:
    plan = objectplan.Plan()
    plan.atoms[("us", ".data")] = list(atoms)
    monkeypatch.setattr(objectplan, "read_dir", lambda files: plan)


def test_audit_reads_the_records_the_symbol_file_and_the_plan(tmp_path, monkeypatch):
    fill = _fill(0x20)
    _project(
        tmp_path, [fill], "D_a = 0x80000010; // type:data rom:0x10 size:0x4\nD_b = 0x80000022; // type:data size:2\n"
    )
    _plan(monkeypatch, Atom(0x80000020, 4, "pad", "drop", 7, "t", "pad_a"))
    (found,) = fillaudit.audit(tmp_path, "FILES.tsv")
    assert (found.key, found.unit, found.path) == ("fill.overlap", "D_b", "versions/us/linker_fill.toml")


def test_audit_with_records_but_no_plan_refuses_instead_of_trusting_them(tmp_path):
    _project(tmp_path, [_fill(0x20)], "")
    (found,) = fillaudit.audit(tmp_path, None)
    assert found.key == "fill.plan" and found.versions == ("us",)


def test_audit_of_a_project_with_no_records_asks_for_nothing(tmp_path):
    (tmp_path / "config.toml").write_text("[version.us]\n")
    assert fillaudit.audit(tmp_path, None) == []


def test_the_pads_of_a_plan_are_its_pad_atoms_of_one_version_only():
    plan = objectplan.Plan()
    plan.atoms[("us", ".data")] = [
        Atom(0x80000020, 4, "pad", "drop", 7, "t", "p"),
        Atom(0x80000030, 4, "func", "keep", 7, "t", "f"),
    ]
    plan.atoms[("eu", ".data")] = [Atom(0x80000040, 4, "pad", "drop", 9, "t", "p")]
    assert fillaudit.pads(plan, "us") == {("7", ".data", 0x80000020, 4)}
