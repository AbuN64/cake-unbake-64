"""Whole-map application keeps one scan/sync and refuses all unsafe members before writing."""

from unittest.mock import Mock

import pytest

from unbake import folders, mergecheck
from unbake.contracts import Config, FileFacts, Refusal


def prepared(cfg: Config, monkeypatch, count=2):
    root = cfg.project.root
    cfg.project.module_map = None
    rows = []
    for i in range(count):
        for j in range(2):
            path = f"src/group{i}/part{j}.c"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(f"int function_{i}_{j}(void) {{ return {j}; }}\n")
            rows.append((path, f"src/group{i}.c"))
    map_file = root / "modules.tsv"
    map_file.write_text("unit\tmodule\n" + "".join(f"{u}\t{m}\n" for u, m in rows))
    verdicts = [{"module": f"src/group{i}.c", "hard": [], "soft": [], "verdict": "merge"} for i in range(count)]
    scan = Mock(return_value=(verdicts, rows, {}))
    monkeypatch.setattr(folders, "_verdicts", scan)
    tracked = Mock(side_effect=lambda _: [str(p.relative_to(root)) for p in root.rglob("*.c")])
    monkeypatch.setattr(folders, "_tracked", tracked)
    stage = Mock()
    monkeypatch.setattr(folders, "_git", stage)
    monkeypatch.setattr(folders.configuration, "load_project", Mock(return_value=cfg.project))
    sync = Mock(return_value={"written": ["Makefile"]})
    monkeypatch.setattr(folders.repo, "sync", sync)
    prove = Mock(return_value={"checked": True})
    monkeypatch.setattr(folders, "prove", prove)
    return map_file, scan, stage, sync, prove


def test_full_map_scans_stages_syncs_and_proves_once(cfg, monkeypatch):
    path, scan, stage, sync, prove = prepared(cfg, monkeypatch, 3)
    result = folders.merge(cfg, {"modules": path, "prove": True})
    assert len(result["modules"]) == 3
    assert [r["units"] for r in result["modules"]] == [2, 2, 2]
    for call in (scan, stage, sync, prove):
        assert call.call_count == 1
    for i in range(3):
        merged = cfg.project.root / f"src/group{i}/group{i}.c"
        assert f"function_{i}_0" in merged.read_text() and f"function_{i}_1" in merged.read_text()
        assert not merged.with_name("part0.c").exists()
    rows = mergecheck.read_modules(path.read_text())
    assert len(rows) == 3
    scan.return_value = ([], rows, {})
    again = folders.merge(cfg, {"modules": path})
    assert again["modules"] == [] and again["kept"] == 3
    assert sync.call_count == stage.call_count == 1


def test_multiple_hard_refusals_leave_all_files_and_map_unchanged(cfg, monkeypatch):
    path, scan, stage, sync, _ = prepared(cfg, monkeypatch)
    verdicts, rows, _ = scan.return_value
    verdicts[0]["hard"] = ["order: different link order"]
    verdicts[1]["hard"] = ["flags: different compiler"]
    before = {u: (cfg.project.root / u).read_bytes() for u, _ in rows}
    old_map = path.read_bytes()
    with pytest.raises(Refusal) as caught:
        folders.merge(cfg, {"modules": path})
    assert len(caught.value.findings[0].missing) == 2
    assert "order:" in caught.value.findings[0].missing[0]
    assert "flags:" in caught.value.findings[0].missing[1]
    assert path.read_bytes() == old_map
    assert {u: (cfg.project.root / u).read_bytes() for u in before} == before
    stage.assert_not_called()
    sync.assert_not_called()


@pytest.mark.parametrize("defect", ["unknown", "duplicate", "missing", "occupied", "empty"])
def test_invalid_selection_never_writes(cfg, monkeypatch, defect):
    path, scan, stage, sync, _ = prepared(cfg, monkeypatch)
    _, rows, _ = scan.return_value
    params = {"modules": path}
    if defect == "unknown":
        params["module"] = "unknown"
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "missing":
        (cfg.project.root / rows[-1][0]).unlink()
    elif defect == "occupied":
        (cfg.project.root / "src/group1/group1.c").write_text("int foreign;\n")
    else:
        scan.return_value = ([], [], {})
    before = path.read_bytes()
    with pytest.raises(Refusal):
        folders.merge(cfg, params)
    assert path.read_bytes() == before
    stage.assert_not_called()
    sync.assert_not_called()


def test_type_conflicts_with_proof_refuse_before_any_mutation(cfg, monkeypatch):
    path, _, stage, sync, prove = prepared(cfg, monkeypatch)
    root = cfg.project.root
    (root / "src/group1/part0.c").write_text("extern int shared;\n")
    (root / "src/group1/part1.c").write_text("extern float shared;\n")
    with pytest.raises(Refusal):
        folders.merge(cfg, {"modules": path, "prove": True})
    assert (root / "src/group0/part0.c").is_file()
    stage.assert_not_called()
    sync.assert_not_called()
    prove.assert_not_called()


def test_merge_carries_configured_map_preserving_original_row_evidence(cfg, monkeypatch):
    path, _, _, _, _ = prepared(cfg, monkeypatch)
    cfg.project.module_map = "ownership.tsv"
    authority = cfg.project.root / cfg.project.module_map
    authority.write_text(
        "unit\tmodule\theader\tevidence\nsrc/group0/part0.c\tsrc/group0.c\tinclude/group.h\tpart0.c original\n"
    )
    result = folders.merge(cfg, {"modules": path, "module": "src/group0.c"})
    assert (
        authority.read_text().splitlines()[1] == "src/group0/group0.c\tsrc/group0.c\tinclude/group.h\tpart0.c original"
    )
    assert "ownership.tsv" in result["written"]


def test_single_module_uses_same_batch_path(cfg, monkeypatch):
    path, scan, stage, sync, _ = prepared(cfg, monkeypatch)
    result = folders.merge(cfg, {"modules": path, "module": "src/group0.c"})
    assert [r["module"] for r in result["modules"]] == ["src/group0.c"]
    assert (cfg.project.root / "src/group1/part0.c").is_file()
    assert scan.call_count == stage.call_count == sync.call_count == 1


@pytest.mark.parametrize("path", ["/outside.c", "../outside.c", "src/../outside.c", "src//bad.c", ".", ".."])
def test_noncanonical_paths_refuse_before_source_reads(cfg, monkeypatch, path):
    map_file, scan, stage, sync, _ = prepared(cfg, monkeypatch)
    _, rows, _ = scan.return_value
    rows[0] = (rows[0][0], path)
    with pytest.raises(Refusal, match="invalid module rows"):
        folders.merge(cfg, {"modules": map_file})
    stage.assert_not_called()
    sync.assert_not_called()


@pytest.mark.parametrize(
    "facts",
    [FileFacts(toolchain="other"), FileFacts(cflags=("-O1",)), FileFacts(omit=("-O2",)), FileFacts(hilo_nops=True)],
)
def test_destination_must_preserve_compiler_facts(cfg, monkeypatch, facts):
    path, _, stage, sync, _ = prepared(cfg, monkeypatch)
    cfg.project.files["src/group0/part0.c"] = facts
    with pytest.raises(Refusal, match="merge refusals"):
        folders.merge(cfg, {"modules": path})
    stage.assert_not_called()
    sync.assert_not_called()


def test_flat_library_ownership_cannot_disappear(cfg, monkeypatch):
    path, _, stage, sync, _ = prepared(cfg, monkeypatch)
    cfg.project.libraries = ("src/group0/part0.c",)
    with pytest.raises(Refusal, match="merge refusals"):
        folders.merge(cfg, {"modules": path})
    stage.assert_not_called()
    sync.assert_not_called()


def test_verdict_uses_same_emitted_order_as_merge_on_real_barrel_rows(cfg, monkeypatch):
    # The reviewed membership TSV sorted filenames; the linked code has the reverse order.
    one = "src/game/AI/BARREL/BARREL_GETSIGNEDANGLETOPOINT.CPP"
    two = "src/game/AI/BARREL/BARREL_UPDATETRACKGOALS.CPP"
    module = "src/game/AI/BARREL.CPP"
    rows = [(one, module), (two, module)]
    seq = [{"o": ".text", "k": "entry", "u": u, "cls": "text", "n": 8} for u in (two, one)]
    monkeypatch.setattr(folders, "_module_facts", Mock(return_value=(rows, {"us": seq}, {})))
    verdicts, original, seqs = folders._verdicts(cfg, {})
    assert original == rows
    assert mergecheck.code_order([one, two], seqs) == [two, one]
    assert verdicts[0]["hard"] == []
    # Another version that requires the contrary order still refuses; TSV order cannot hide it.
    seqs["de"] = list(reversed(seq))
    verdicts, _, _ = folders._verdicts(cfg, {})
    assert any(reason.startswith("order: de text") for reason in verdicts[0]["hard"])
