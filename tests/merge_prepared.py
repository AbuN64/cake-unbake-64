"""The shared setup of the merge tests: a project of `count` modules of two units each, the scan, git, sync and prove
seams of `folders` replaced by mocks."""

from __future__ import annotations

from unittest.mock import Mock

from unbake import folders
from unbake.contracts import Config


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
