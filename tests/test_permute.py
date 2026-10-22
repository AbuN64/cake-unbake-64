"""permute: finding and replacing a function's text, and the permuter run end to end with a stand-in permuter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unbake import permute
from unbake.contracts import Refusal

SOURCE = "void f(void);\nint g(int a)\n{\n  if (a) { return 1; }\n  return 0;\n}\nvoid f(void) { g(1); }\n"


def test_span_finds_the_definition_not_the_prototype_and_matches_braces() -> None:
    start, end = permute.span(SOURCE, "g") or (0, 0)
    assert SOURCE[start:end].startswith("int g(int a)") and SOURCE[start:end].endswith("return 0;\n}")
    assert permute.span(SOURCE, "f") is not None and permute.span(SOURCE, "h") is None


def test_replace_swaps_one_definition_and_refuses_a_missing_one() -> None:
    out = permute.replace(SOURCE, "g", "int g(int a) { return a; }")
    assert "return a;" in out and "return 0;" not in out and out.endswith("void f(void) { g(1); }\n")
    with pytest.raises(Refusal):
        permute.replace(SOURCE, "h", "x")


def test_search_runs_the_permuter_and_returns_the_best_body(tmp_path: Path) -> None:
    fake = tmp_path / "permuter"
    fake.write_text(
        '#!/bin/sh\nfor d in "$@"; do :; done\n'
        'mkdir "$d/output-5-1" "$d/output-9-1"\n'
        'printf 5 > "$d/output-5-1/score.txt"; printf 9 > "$d/output-9-1/score.txt"\n'
        'printf "int g(int a) { return 7; }\\n" > "$d/output-5-1/source.c"\n'
        'printf "int g(int a) { return 8; }\\n" > "$d/output-9-1/source.c"\n'
        'echo "base score = 20"\n'
    )
    fake.chmod(0o755)
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path), host=SimpleNamespace(workers=2))
    ctx = {"source": SOURCE, "script": "true", "assemble": ["/usr/bin/touch", "{out}"]}
    work = tmp_path / "work"
    found = permute.search(config, "g", ctx, {"permute": 30, "permuter": str(fake)}, work)  # type: ignore[arg-type]
    assert (found["base_score"], found["best_score"], found["outputs"]) == (20, 5, 2)
    assert "return 7" in found["function"]
    assert permute.candidate(SOURCE, "g", found) is not None and "return 7" in (
        permute.candidate(SOURCE, "g", found) or ""
    )
    assert (work / "permute/compile.sh").read_text().startswith("#!/bin/sh\nset -- ")


def test_search_without_a_permuter_names_how_to_pass_one(tmp_path: Path) -> None:
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path), host=SimpleNamespace(workers=2))
    with pytest.raises(Refusal) as refused:
        permute.search(config, "g", {}, {"permute": 5}, tmp_path)  # type: ignore[arg-type]
    assert "UNBAKE_PERMUTER" in refused.value.findings[0].reason
