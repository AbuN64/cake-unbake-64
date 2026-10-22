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
    ctx = {"source": SOURCE, "script": "true", "assemble": ["/usr/bin/touch", "{out}"], "asm_path": "nonmatchings/g.s"}
    work = tmp_path / "work"
    found = permute.search(config, "g", ctx, {"permute": 30, "permuter": str(fake)}, work)  # type: ignore[arg-type]
    assert (found["base_score"], found["best_score"], found["outputs"]) == (20, 5, 2)
    assert "return 7" in found["function"]
    assert permute.candidate(SOURCE, "g", found) is not None and "return 7" in (
        permute.candidate(SOURCE, "g", found) or ""
    )
    assert (work / "permute/compile.sh").read_text().startswith("#!/bin/sh\nset -- ")
    assert "perm_add_mask = 0" in (work / "permute/settings.toml").read_text()


def test_search_without_a_permuter_names_how_to_pass_one(tmp_path: Path) -> None:
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path), host=SimpleNamespace(workers=2))
    with pytest.raises(Refusal) as refused:
        permute.search(config, "g", {}, {"permute": 5}, tmp_path)  # type: ignore[arg-type]
    assert "UNBAKE_PERMUTER" in refused.value.findings[0].reason


def test_the_permuter_runs_under_the_venv_python_it_needs_not_the_shebangs(tmp_path: Path, monkeypatch) -> None:
    venv = tmp_path / "venv-permuter/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("#!/bin/sh\n")
    (venv / "python").chmod(0o755)
    script = tmp_path / "permuter.py"
    script.write_text("")
    monkeypatch.delenv("UNBAKE_PERMUTER_PYTHON", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert permute._python(str(script)) == [str(venv / "python")]
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "none"))
    assert permute._python(str(script)) == []


def _proc(root: Path, pid: int, parent: int, command: str) -> None:
    folder = root / str(pid)
    folder.mkdir(parents=True)
    (folder / "cmdline").write_bytes(command.replace(" ", "\0").encode())
    (folder / "stat").write_text(f"{pid} (python3) S {parent} 1 1")


def test_running_counts_runs_not_their_workers_and_places_are_capped(tmp_path: Path, monkeypatch) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc(proc, 100, 1, "python3 /x/permuter.py -j 6 dir")
    _proc(proc, 101, 100, "python3 /x/permuter.py -j 6 dir")  # a worker of the first
    _proc(proc, 200, 1, "/x/venv/bin/python /x/permuter.py -j 4 other")
    _proc(proc, 300, 1, "vim permuter.py")  # not a run
    assert [p for p, _ in permute._running(proc, tmp_path / "none")] == [100, 200]
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(permute, "QUEUE_S", 0.0)
    cfg = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    from unbake import store

    monkeypatch.setattr(store, "exclusive", lambda *a, **k: __import__("contextlib").nullcontext(True))
    with pytest.raises(Refusal) as refused:
        permute._place(cfg, proc)
    assert refused.value.findings[0].key == "permute.cap" and "pid 100" in refused.value.findings[0].reason
    monkeypatch.setenv("UNBAKE_PERMUTER_CAP", "3")
    mark = permute._place(cfg, proc)
    assert mark.is_file()
    mark.unlink()


def test_shield_hides_what_the_permuters_parser_misreads_and_the_script_puts_it_back() -> None:
    source = (
        "struct A { int x; } __attribute__((aligned(4)));\n"
        "typedef struct { char pad[0x11D8 - sizeof(Head) - sizeof(Cam)]; } B;\n"
        "typedef char C_size_check[(sizeof(C) == 0x8) ? 1 : -1];\n"
    )
    text, defines, named = permute._shield(source)
    assert named == []
    assert "__attribute__" not in text and "UNBAKE_ATTR_0" in text
    assert "[0x11D8 - UNBAKE_SIZEOF_Head - UNBAKE_SIZEOF_Cam]" in text
    assert "UNBAKE_SIZEOF_C" in text  # a bracket holding a minus is shielded whole
    assert "-DUNBAKE_ATTR_0=__attribute__((aligned(4)))" in defines and "-DUNBAKE_SIZEOF_Head=sizeof(Head)" in defines
    script = permute._script(Path("/p"), '#!/bin/sh\ncpp -P "$1" > "$3.i"\n', defines)
    assert "'-DUNBAKE_ATTR_0=__attribute__((aligned(4)))' " in script and script.rindex("-DUNBAKE") < script.rindex(
        '"$1"'
    )


def test_a_run_is_kept_under_the_cpu_cap_and_says_how_to_extend(monkeypatch) -> None:
    monkeypatch.delenv("UNBAKE_PERMUTE_CPU", raising=False)
    assert permute._budget(30, 8) == (30, 8, "") and permute._budget(60, 4)[2] == ""  # 240 CPU s
    wall, jobs, note = permute._budget(1200, 8)  # the 20 min x 8 jobs lane runs
    assert wall * jobs <= 300 and jobs == 1 and "UNBAKE_PERMUTE_CPU" in note
    wall, jobs, note = permute._budget(100, 8)
    assert (wall, jobs) == (100, 3) and "100 s x 3 jobs" in note
    monkeypatch.setenv("UNBAKE_PERMUTE_CPU", "9000")
    assert permute._budget(1200, 6) == (1200, 6, "")


def test_a_permuter_that_prints_no_base_score_fails_fast_with_its_own_error(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "permuter"
    fake.write_text('#!/bin/sh\necho "Traceback: cannot compile the base" >&2\nexit 1\n')
    fake.chmod(0o755)
    monkeypatch.setattr(permute, "PREFLIGHT", 5.0)
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path), host=SimpleNamespace(workers=2))
    ctx = {"source": SOURCE, "script": "true", "assemble": ["/usr/bin/touch", "{out}"], "asm_path": "nonmatchings/g.s"}
    with pytest.raises(Refusal) as raised:
        permute.search(config, "g", ctx, {"permute": 300, "permuter": str(fake)}, tmp_path / "work")  # type: ignore[arg-type]
    assert "cannot compile the base" in str(raised.value.findings[0])


def test_a_typedef_attribute_the_parser_cannot_read_leaves_the_text_and_the_script_writes_it_back(tmp_path) -> None:
    source = "typedef struct {\n  int a;\n} __attribute__((aligned(4))) SoundEnvelope;\nint g(void) { return 0; }\n"
    text, defines, named = permute._shield(source)
    assert "__attribute__" not in text and "} SoundEnvelope;" in text and defines == []
    assert named == [("SoundEnvelope", "__attribute__((aligned(4)))")]
    script = permute._script(tmp_path, '#!/bin/sh\ncat "$1" > "$3"\n', defines, named)
    (tmp_path / "base.c").write_text(text)
    import subprocess

    done = subprocess.run(
        ["sh", "-c", script, "x", str(tmp_path / "base.c"), "-o", str(tmp_path / "out.c")], check=False
    )
    assert done.returncode == 0
    assert "} __attribute__((aligned(4))) SoundEnvelope;" in (tmp_path / "out.c").read_text()
