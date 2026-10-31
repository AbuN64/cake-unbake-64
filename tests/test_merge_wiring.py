"""Merge writes at a named target, takes one unit as a rename, regenerates the fill rows and emits classes in their own
order; split rewrites the module maps; the pure rules under them."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from merge_prepared import prepared

from unbake import classwise, folders, linkfill, merge, mergecheck
from unbake import merge as unitmerge
from unbake.contracts import Refusal


def test_the_target_column_names_the_merged_file_and_a_module_without_one_keeps_the_folder_form() -> None:
    text = "unit\tmodule\ttarget\na.c\tM\tsrc/M.c\nb.c\tM\tsrc/M.c\nc.c\tN\t\n"
    assert mergecheck.read_targets(text) == {"M": "src/M.c"}
    assert mergecheck.merged_path("src/N.c", None) == "src/N/N.c"
    assert mergecheck.merged_path("src/M.c", "src/M.c") == "src/M.c"
    assert mergecheck.read_targets("unit\tmodule\na.c\tM\n") == {}
    with pytest.raises(ValueError, match="two targets"):
        mergecheck.read_targets("unit\tmodule\ttarget\na.c\tM\tx.c\nb.c\tM\ty.c\n")


def test_merge_writes_the_file_at_the_named_target(cfg, monkeypatch) -> None:
    path, *_ = prepared(cfg, monkeypatch, 1)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([f"{lines[0]}\ttarget", *(f"{line}\tsrc/group0.c" for line in lines[1:])]) + "\n")
    folders.merge(cfg, {"modules": path})
    root = cfg.project.root
    assert "function_0_0" in (root / "src/group0.c").read_text()
    assert not (root / "src/group0/group0.c").exists()
    assert not (root / "src/group0/part0.c").exists()


def test_a_module_of_one_unit_is_a_rename_to_its_target(cfg, monkeypatch) -> None:
    path, scan, _, sync, _ = prepared(cfg, monkeypatch, 1)
    root = cfg.project.root
    path.write_text("unit\tmodule\ttarget\nsrc/group0/part0.c\tsrc/one.c\tsrc/one.c\n")
    scan.return_value = ([], [("src/group0/part0.c", "src/one.c")], {})
    result = folders.merge(cfg, {"modules": path})
    assert result["modules"][0]["units"] == 1
    assert "function_0_0" in (root / "src/one.c").read_text()
    assert not (root / "src/group0/part0.c").exists()
    assert path.read_text().splitlines()[1].startswith("src/one.c\t")
    assert sync.call_count == 1


def span_seq() -> list[dict]:
    one = {"o": ".rodata", "k": "entry", "u": "a", "cls": "rodata", "n": 4, "parts": {"rodata": [(4, 4)]}}
    two = {"o": ".rodata", "k": "entry", "u": "b", "cls": "rodata", "n": 4, "parts": {"rodata": [(4, 4)]}}
    return [one, {"o": ".rodata", "k": "align", "a": 8}, two]


def test_refill_turns_the_bytes_a_merged_object_leaves_short_into_a_row_and_drops_the_rows_inside() -> None:
    rom = bytes(0x100) + bytes(8) + b"\x00\x12\x00\x34" + bytes(8)
    inside = linkfill.Fill("old", ".rodata", 0x104, 0x80000004, b"\x01", "x")
    outside = linkfill.Fill("keep", ".rodata", 0x120, 0x80000020, b"\x02", "x")
    rows, dropped, written = linkfill.refill(
        [outside, inside], [("M", ".rodata", 0x100, 0x108, 0x10C, 0x80000000)], rom
    )
    assert (dropped, written) == (1, 1)
    assert [(f.object, f.rom, f.address, f.data) for f in rows] == [
        ("M", 0x108, 0x80000008, b"\x00\x12\x00\x34"),
        ("keep", 0x120, 0x80000020, b"\x02"),
    ]
    zeros, dropped, written = linkfill.refill([], [("M", ".rodata", 0x100, 0x108, 0x10C, 0x80000000)], bytes(0x200))
    assert zeros == [] and written == 0


def test_merge_regenerates_the_fill_file_of_every_version(cfg, monkeypatch) -> None:
    root = cfg.project.root
    version = cfg.project.versions[0]
    monkeypatch.setattr(folders.rommap, "pieces", Mock(return_value=(("rodata", 0x100, 0x200, 0x80000000),)))
    rom = bytearray(0x200)
    rom[0x108:0x10C] = b"\x00\x12\x00\x34"
    monkeypatch.setattr(folders.configuration, "rom", Mock(return_value=Mock(read_bytes=lambda: bytes(rom))))
    monkeypatch.setattr(folders.store, "write", lambda target, data: target.write_bytes(data))
    out = folders._regenerate_fills(cfg, {version: span_seq()}, [("M", ["a", "b"], "src/M.c", {})])
    assert out == {version: {"dropped": 0, "written": 1}}
    rows = linkfill.read(root, version)
    assert [(f.object, f.rom, f.address) for f in rows] == [("src/M.c", 0x108, 0x80000008)]
    # the next merge finds the rows it wrote inside the span of a merged object that now fills it
    rom[0x108:0x10C] = bytes(4)
    again = folders._regenerate_fills(cfg, {version: span_seq()}, [("M", ["a", "b"], "src/M.c", {})])
    assert again == {version: {"dropped": 1, "written": 0}}
    assert not (root / linkfill.path(version)).exists()


def test_a_module_of_one_unit_has_no_fill_to_regenerate(cfg, monkeypatch) -> None:
    pieces = Mock()
    monkeypatch.setattr(folders.rommap, "pieces", pieces)
    version = cfg.project.versions[0]
    assert folders._regenerate_fills(cfg, {version: span_seq()}, [("M", ["a"], "src/M.c", {})]) == {}


A = "int shared;\nint first_text(void) { return shared; }\nint first_data = 1;\n"
B = "int second_text(void) { return first_text(); }\nint second_data = 2;\n"


def test_classes_linked_in_their_own_unit_order_are_emitted_in_that_order() -> None:
    classes = {
        "a": {"first_text": "text", "first_data": "data", "shared": "bss"},
        "b": {"second_text": "text", "second_data": "data"},
    }
    orders = {"text": {"us": ["a", "b"]}, "data": {"us": ["b", "a"]}, "bss": {"us": ["a"]}}
    text, conflicts = classwise.arrange(
        [("a", A, unitmerge.mask(A), unitmerge.decls(A, "a")), ("b", B, unitmerge.mask(B), unitmerge.decls(B, "b"))],
        orders,
        classes,
    )
    assert conflicts == []
    assert text.index("first_text") < text.index("second_text")
    assert text.index("second_data") < text.index("first_data")


def test_merge_keeps_plain_concatenation_when_every_class_follows_the_unit_order() -> None:
    kept = merge.merge(
        [("a", A), ("b", B)], lambda *_: False, orders={"text": {"us": ["a", "b"]}, "data": {"us": ["a", "b"]}}
    )
    assert kept["text"].index("first_data") < kept["text"].index("second_text")
    swapped = merge.merge(
        [("a", A), ("b", B)],
        lambda *_: False,
        orders={"text": {"us": ["a", "b"]}, "data": {"us": ["b", "a"]}},
        classes={
            "a": {"first_text": "text", "first_data": "data", "shared": "bss"},
            "b": {"second_text": "text", "second_data": "data"},
        },
    )
    assert swapped["conflicts"] == []
    assert swapped["text"].index("second_data") < swapped["text"].index("first_data")


def test_a_split_unit_replaces_its_map_row_with_a_row_per_piece() -> None:
    text = "unit\tmodule\tnote\n# c\nu.c\tOLD\tkeep\nz.c\tZ\t\n"
    out = mergecheck.split_modules(text, "u.c", {"cam": "src/cam/u.c", "ply": "src/ply/u.c"})
    assert out.splitlines() == [
        "unit\tmodule\tnote",
        "# c",
        "src/cam/u.c\tcam\tkeep",
        "src/ply/u.c\tply\tkeep",
        "z.c\tZ\t",
    ]


def test_split_writes_one_file_per_target_and_rewrites_the_module_map(cfg, monkeypatch) -> None:
    root = cfg.project.root
    cfg.project.module_map = None
    unit = "src/u.c"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / unit).write_text("int cam_fn(void) { return 1; }\nint ply_fn(void) { return 2; }\n")
    assign = root / "assign.tsv"
    assign.write_text(
        "symbol\ttarget\tcurrent_src\ncam_fn\tsrc/CAM.c\tsrc/u.c\nply_fn\tsrc/PLY.c\tsrc/u.c\nother\tsrc/X.c\tv.c\n"
    )
    modules = root / "modules.tsv"
    modules.write_text(f"unit\tmodule\n{unit}\told\n")
    monkeypatch.setattr(folders, "_tracked", Mock(return_value=[unit]))
    monkeypatch.setattr(folders, "_git", Mock())
    monkeypatch.setattr(folders.configuration, "load_project", Mock(return_value=cfg.project))
    sync = Mock(return_value={"written": ["Makefile"]})
    monkeypatch.setattr(folders.repo, "sync", sync)
    result = folders.split(cfg, {"file": unit, "assign": assign, "modules": modules})
    assert result["pieces"] == {"src/CAM.c": "src/CAM/u.c", "src/PLY.c": "src/PLY/u.c"}
    assert (root / "src/CAM/u.c").read_text() == "int cam_fn(void) { return 1; }\n"
    assert (root / "src/PLY/u.c").read_text() == "int ply_fn(void) { return 2; }\n"
    assert not (root / unit).exists()
    assert modules.read_text().splitlines() == ["unit\tmodule", "src/CAM/u.c\tsrc/CAM.c", "src/PLY/u.c\tsrc/PLY.c"]
    sync.assert_called_once()


def test_split_refuses_a_unit_of_one_target_and_a_symbol_of_two(cfg, monkeypatch) -> None:
    root = cfg.project.root
    unit = "src/u.c"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / unit).write_text("int a(void) { return 1; }\n")
    monkeypatch.setattr(folders, "_tracked", Mock(return_value=[unit]))
    assign = root / "assign.tsv"
    assign.write_text("symbol\ttarget\na\tsrc/X.c\n")
    with pytest.raises(Refusal, match="belongs to 1 target"):
        folders.split(cfg, {"file": unit, "assign": assign})
    assign.write_text("version\tsymbol\ttarget\nus\ta\tsrc/X.c\neu\ta\tsrc/Y.c\n")
    with pytest.raises(Refusal, match="more than one target"):
        folders.split(cfg, {"file": unit, "assign": assign})
    assert (root / unit).read_text() == "int a(void) { return 1; }\n"
