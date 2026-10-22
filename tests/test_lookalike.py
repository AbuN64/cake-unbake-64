"""lookalike: a padded view is matched to existing records of the same size and named-field offsets."""

from __future__ import annotations

from pathlib import Path

from unbake import lookalike

REAL = (
    "typedef struct Real {\n    u32 flags;\n    char pad4[0x14];\n    void *impact;\n    char pad1C[0x20];\n} Real;\n"
)
VIEW = (
    "typedef struct View {\n    u32 flags;\n    char pad4[0x14];\n    void *impact;\n    char pad1C[0x20];\n} View;\n"
)
OTHER = "typedef struct Other {\n    u32 a;\n    u32 b;\n    u32 c;\n} Other;\n"


def test_a_padded_view_names_the_record_of_its_size_and_offsets(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "src/real.c").write_text(REAL)
    (tmp_path / "include/other.H").write_text(OTHER)
    (tmp_path / "src/view.c").write_text(VIEW)
    lookalike._cache.clear()
    said = lookalike.hint(tmp_path, "src/view.c", VIEW, 1)
    assert "Real (src/real.c)" in said and "Other" not in said and "View (" not in said
    assert lookalike.hint(tmp_path, "src/view.c", OTHER, 1) == ""  # no padded field named: nothing to match
