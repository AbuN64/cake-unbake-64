"""The linker's alignment fill: the record file, the link script and splat lines it makes, and its generator over mocked
seams (the object plan, the ROM, the version's pieces)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unbake import build, linkfill, objectplan
from unbake.contracts import Refusal, Segment
from unbake.linkfill import Fill
from unbake.objectplan import Atom
from unbake.tree import Module, Tree

TEXT = """# generated
version = "us"

[[fill]]
object = "7"
section = ".game_data"
rom = 0x1018
address = 0x80200018
size = 4
bytes = "75726500"
provenance = "unattributed"
"""


def _row(**change: object) -> str:
    row = {
        "object": '"7"',
        "section": '".data"',
        "rom": "0x10",
        "address": "0x80000010",
        "size": "2",
        "bytes": '"0102"',
    }
    row |= {"provenance": '"unattributed"'} | {k: v for k, v in change.items() if v is not None}
    for gone in (k for k, v in change.items() if v is None):
        del row[gone]
    return "[[fill]]\n" + "".join(f"{k} = {v}\n" for k, v in row.items())


def test_a_fill_file_round_trips_through_render_and_parse():
    fills = [
        Fill("9", ".data", 0x40, 0x80000040, b"\x01\x02"),
        Fill("7", ".data", 0x10, 0x80000010, b"\xaa", "earlier-link 0x8"),
    ]
    text = linkfill.render("us", fills, "generated")
    assert linkfill.parse(text, "us") == sorted(fills, key=lambda f: f.rom)
    assert text.startswith('# generated\nversion = "us"\n')


def test_parse_reads_the_rows_of_a_file():
    (fill,) = linkfill.parse(TEXT, "us")
    assert (fill.object, fill.section, fill.rom, fill.address, fill.data, fill.size) == (
        "7", ".game_data", 0x1018, 0x80200018, b"ure\x00", 4
    )  # fmt: skip
    assert fill.provenance == "unattributed"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('version = "de"\n', "the file is for de"),
        (_row(provenance=None), "wrong fields: provenance"),
        (_row(extra="1"), "wrong fields: extra"),
        (_row(bytes='"0z"'), "bytes is not hex"),
        (_row(size="3"), "size 3 is not the 2 nonzero bytes"),
        (_row(bytes='"0000"'), "size 2 is not the 2 nonzero bytes"),
        (_row() + _row(rom="0x11"), "overlap"),
    ],
)
def test_parse_refuses_a_file_that_is_not_the_schema(text: str, reason: str):
    with pytest.raises(Refusal) as refused:
        linkfill.parse(f'version = "us"\n{text}' if not text.startswith("version") else text, "us")
    assert reason in refused.value.findings[0].reason
    assert refused.value.findings[0].key == "fill.record"


def test_a_missing_file_has_no_fill(tmp_path: Path):
    assert linkfill.read(tmp_path, "us") == []
    (tmp_path / linkfill.path("us")).parent.mkdir(parents=True)
    (tmp_path / linkfill.path("us")).write_text(TEXT)
    assert [f.rom for f in linkfill.read(tmp_path, "us")] == [0x1018]


def test_statement_moves_to_the_bytes_and_emits_each():
    assert linkfill.statement(Fill("7", ".data", 0x1018, 0, b"\x01\xff"), 0x1000) == ". = 0x18; BYTE(0x01); BYTE(0xFF)"


def test_piece_is_a_data_cell_of_its_own_in_the_segment():
    assert linkfill.piece(Fill("7", ".data", 0x1018, 0, b"\x01"), "main") == (0x1018, "data", "main/fill_1018")


def test_rom_of_wants_exactly_one_piece():
    pieces = [("a", 0x1000, 0x1100, 0x80000000), ("b", 0x2000, 0x2100, 0x80000080)]
    assert linkfill.rom_of(pieces, 0x80000010) == 0x1010
    assert linkfill.rom_of(pieces, 0x80000090) is None  # in both: ambiguous
    assert linkfill.rom_of(pieces, 0x70000000) is None


def test_earlier_link_names_a_single_place_of_the_bytes():
    windows = [(0x100, b"\x00\x01\x02\x03\x01\x02"), (0x400, b"\x09")]
    assert linkfill.earlier_link(b"\x03\x01", windows) == "earlier-link 0x103"
    assert linkfill.earlier_link(b"\x01\x02", windows) == linkfill.UNATTRIBUTED  # twice
    assert linkfill.earlier_link(b"\x07", windows) == linkfill.UNATTRIBUTED


def test_a_zero_gap_gets_no_record():
    rom = bytes(8) + b"\x01\x02" + bytes(6)
    fills = linkfill.build([("1", ".data", 0x80000000, 0, 8), ("2", ".data", 0x80000008, 8, 2)], rom, [])
    assert [(f.object, f.rom, f.data) for f in fills] == [("2", 8, b"\x01\x02")]


def _config(root: Path) -> SimpleNamespace:
    segments = (
        Segment("main", 0x1000, 0x80000000, "code"),
        Segment("window", 0x1020, 0x90000000, "code", "earlier_build", "main"),
    )
    version = SimpleNamespace(segments=segments)
    project = SimpleNamespace(root=root, versions=("us", "eu"), version={"us": version, "eu": version})
    return SimpleNamespace(project=project)


def _plan(monkeypatch, atoms: list[Atom]) -> None:
    plan = objectplan.Plan()
    plan.atoms[("us", ".data")] = atoms
    plan.atoms[("eu", ".data")] = [Atom(0x80000030, 4, "func", "keep", 1, "t", "f")]  # no pad atom
    monkeypatch.setattr(objectplan, "read_dir", lambda files: plan)


@pytest.fixture
def plan_dir(tmp_path: Path) -> Path:
    for name in ("FILES.tsv", "OBJECTS.tsv", "ASSIGN.tsv"):
        (tmp_path / name).touch()
    return tmp_path


def test_generate_writes_a_file_for_the_versions_with_nonzero_pad_atoms(plan_dir: Path, tmp_path: Path, monkeypatch):
    root = tmp_path / "project"
    for v in ("us", "eu"):
        (root / "versions" / v).mkdir(parents=True)
    (root / linkfill.path("eu")).write_text(TEXT)  # a file from an earlier run
    rom = bytearray(0x2000)
    rom[0x1008:0x100C] = b"\x11\x22\x33\x44"
    rom[0x1020:0x1024] = b"\x11\x22\x33\x44"  # the same bytes once in the earlier link's window
    config = _config(root)
    _plan(
        monkeypatch,
        [Atom(0x80000008, 4, "pad", "drop", 3, "t", "pad_a"), Atom(0x80000010, 4, "pad", "drop", 4, "t", "z")],
    )
    monkeypatch.setattr(
        linkfill.versions,
        "pieces",
        lambda c, v: (("main", 0x1000, 0x1020, 0x80000000), ("window", 0x1020, 0x1100, 0x90000000)),
    )
    monkeypatch.setattr(
        linkfill.configuration, "rom", lambda project, v: SimpleNamespace(read_bytes=lambda: bytes(rom))
    )
    out = linkfill.generate(config, {"plan": str(plan_dir / "FILES.tsv")})
    assert out["versions"] == {"us": {"fills": 1, "bytes": 4}, "eu": {"fills": 0, "bytes": 0}}
    assert out["written"] == [linkfill.path("us")]
    (fill,) = linkfill.read(root, "us")
    assert (fill.object, fill.section, fill.rom, fill.address, fill.provenance) == (
        "3", ".data", 0x1008, 0x80000008, "earlier-link 0x1020"
    )  # fmt: skip
    assert not (root / linkfill.path("eu")).exists()


def test_generate_refuses_a_pad_atom_no_piece_holds(plan_dir: Path, tmp_path: Path, monkeypatch):
    (tmp_path / "project").mkdir()
    _plan(monkeypatch, [Atom(0x70000000, 4, "pad", "drop", 3, "t", "pad_a")])
    monkeypatch.setattr(linkfill.versions, "pieces", lambda c, v: (("main", 0x1000, 0x1020, 0x80000000),))
    monkeypatch.setattr(
        linkfill.configuration, "rom", lambda project, v: SimpleNamespace(read_bytes=lambda: bytes(0x2000))
    )
    with pytest.raises(Refusal) as refused:
        linkfill.generate(_config(tmp_path / "project"), {"plan": str(plan_dir / "FILES.tsv")})
    assert refused.value.findings[0].key == "fill.plan"


def test_generate_wants_the_whole_plan_beside_the_files(tmp_path: Path):
    (tmp_path / "FILES.tsv").touch()
    with pytest.raises(Refusal) as refused:
        linkfill.generate(_config(tmp_path), {"plan": str(tmp_path / "FILES.tsv")})
    assert refused.value.findings[0].key == "fill.plan"


def _tree_config(tmp_path: Path) -> SimpleNamespace:
    from test_build import make_config  # the build tests' project: versions eu-1 and us, one `main` segment

    return make_config(tmp_path)


def _tree() -> Tree:
    def named(path: str, *names: str) -> Module:
        kinds = {n: ".text" for n in names}
        return Module(path, "main", {"us": names, "eu-1": names}, (), {}, kinds)

    table = {v: {f"f_{n}": (0x80200000 + n, 0x1000 + n * 0x10, 0x10, "func") for n in (1, 3)} for v in ("us", "eu-1")}
    return Tree("digest", (named("src/a.c", "f_1"), named("src/b.c", "f_3")), table)


def test_the_link_script_emits_a_fill_between_the_objects_it_lies_between(tmp_path: Path):
    config = _tree_config(tmp_path)
    (tmp_path / linkfill.path("us")).write_text(TEXT)
    lines = [x.strip() for x in build.linker(config, _tree()).decode().splitlines() if "src/" in x or "BYTE" in x]
    assert lines[:3] == [
        "src/a.c.o(.text);",
        ". = 0x18; BYTE(0x75); BYTE(0x72); BYTE(0x65); BYTE(0x00);",
        "src/b.c.o(.text);",
    ]


def test_splat_cuts_a_fill_into_a_data_piece_of_its_own(tmp_path: Path):
    config = _tree_config(tmp_path)
    (tmp_path / linkfill.path("us")).write_text(TEXT)
    assert "main/fill_1018]" in build.splat_yaml(config, "us", _tree()).decode()
    assert "fill_1018" not in build.splat_yaml(config, "eu-1", _tree()).decode()
