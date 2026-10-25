"""The build proof on tiny git trees: make and the object reader are faked, everything else is real."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unbake import build, buildproof, process, store
from unbake.contracts import Refusal

FILES = {
    "include/h.h": "int Thing(void);\n",
    "src/a.c": '#include "h.h"\nint A(void) { return Thing(); }\n',
    "src/b.CPP": "int B(void) { return 2; }\n",
    "src/c.c": "int C(void) { return 3; }\n",
    "README.md": "x\n",
}
VERSIONS = ["us", "de"]


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    root = tmp_path / "repo"
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    _git(root, "init", "-q")
    (root / ".gitignore").write_text("build/\n.unbake/\nlocal.mk\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    config = SimpleNamespace(project=SimpleNamespace(root=root, versions=VERSIONS))
    log: list[list[str]] = []

    def git(_c: Any, *args: str) -> Any:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
        return SimpleNamespace(stdout=out.stdout.decode(), exit=0)

    def fake_build(_c: Any, targets: list[str], _jobs: int) -> tuple[set[str], list[str]]:
        log.append(list(targets))
        absent, bad = set(), []
        for target in targets:
            src = root / target.split("/", 2)[2].removesuffix(".o")
            if not src.is_file() or (target.startswith("build/de/") and src.name == "b.CPP"):
                absent.add(target)  # b.CPP has no rule in de
                continue
            text = src.read_text()
            if "BROKEN" in text:
                bad.append(f"make: *** [Makefile:1: {target}] Error 1")
                continue
            obj = root / target
            obj.parent.mkdir(parents=True, exist_ok=True)
            obj.write_text(" ".join(re.sub(r"/\*.*?\*/", "", text).replace("Renamed", "Thing").split()))
        return absent, bad

    monkeypatch.setattr(process, "git", git)
    monkeypatch.setattr(buildproof, "_build", fake_build)
    monkeypatch.setattr(buildproof, "_shape", lambda p: [p.read_text()])
    monkeypatch.setattr(build, "local_mk", lambda c: b"")
    monkeypatch.setattr(store, "write", lambda p, b: True)
    monkeypatch.setattr(process, "state", lambda r: r / ".unbake")
    return SimpleNamespace(root=root, config=config, log=log)


def _patch(tree: SimpleNamespace, name: str, edit: Any) -> Path:
    edit(tree.root)
    diff = _git(tree.root, "diff", "--no-color")
    added = _git(tree.root, "ls-files", "-o", "--exclude-standard").split()
    if added:
        _git(tree.root, "add", "-N", *added)
        diff = _git(tree.root, "diff", "--no-color")
    path = tree.root.parent / name
    path.write_text(diff)
    _git(tree.root, "reset", "-q", "--hard")
    _git(tree.root, "clean", "-fdq")
    return path


def _run(tree: SimpleNamespace, *patches: Path) -> dict[str, Any]:
    return buildproof.buildproof_command(
        tree.config, {"against": "HEAD", "patch": [str(p) for p in patches], "version": [], "jobs": 1}
    )


def _clean(tree: SimpleNamespace) -> bool:
    import subprocess

    out = subprocess.run(["git", "status", "--porcelain"], cwd=tree.config.project.root, capture_output=True, text=True)
    return out.stdout.strip() == ""
def _keys(refused: pytest.ExceptionInfo[Refusal]) -> set[str]:
    return {f.key for f in refused.value.findings}


def test_a_rename_that_builds_the_same_passes_and_reaches_header_consumers(tree: SimpleNamespace) -> None:
    def edit(root: Path) -> None:
        (root / "include/h.h").write_text("int Renamed(void);\n")
        (root / "src/a.c").write_text('#include "h.h"\nint A(void) { return Renamed(); }\n')

    result = _run(tree, _patch(tree, "ok.diff", edit))
    assert result["compared"] == 2  # a.c in us and de; b.CPP and c.c are not reached
    assert result["identical"] == 2
    assert _clean(tree)
    assert not any("b.CPP" in t for t in tree.log[0])


def test_a_dot_cpp_unit_and_a_dot_c_unit_are_both_proved(tree: SimpleNamespace) -> None:
    def edit(root: Path) -> None:
        (root / "src/b.CPP").write_text("int B(void) { return 2; } /* note */\n")
        (root / "src/c.c").write_text("int C(void) { return 3; } /* note */\n")

    result = _run(tree, _patch(tree, "units.diff", edit))
    assert result["compared"] == 3  # b.CPP in us only (no rule in de), c.c in both
    assert result["absent"] == 1


def test_a_change_that_breaks_make_fails(tree: SimpleNamespace) -> None:
    def edit(root: Path) -> None:
        (root / "src/c.c").write_text("BROKEN\n")

    with pytest.raises(Refusal) as refused:
        _run(tree, _patch(tree, "bad.diff", edit))
    assert "buildproof.make" in _keys(refused)
    assert _clean(tree)


def test_a_real_change_differs(tree: SimpleNamespace) -> None:
    def edit(root: Path) -> None:
        (root / "src/c.c").write_text("int C(void) { return 4; }\n")

    with pytest.raises(Refusal) as refused:
        _run(tree, _patch(tree, "diff.diff", edit))
    assert _keys(refused) == {"buildproof.differs"}
    assert _clean(tree)  # the patch is taken off on a failing proof too


def test_a_unit_that_disappears_from_the_set_fails(tree: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    def edit(root: Path) -> None:
        (root / "src/c.c").write_text("int C(void) { return 3; } /* x */\n")

    patch = _patch(tree, "gone.diff", edit)
    real = buildproof._build
    calls: list[int] = []

    def drop(c: Any, targets: list[str], jobs: int) -> tuple[set[str], list[str]]:
        calls.append(1)
        out = real(c, targets, jobs)
        if len(calls) == 2:  # after the change one object is not built, and make said nothing
            (tree.root / "build/us/src/c.c.o").unlink()
        return out

    monkeypatch.setattr(buildproof, "_build", drop)
    with pytest.raises(Refusal) as refused:
        _run(tree, patch)
    assert "buildproof.set" in _keys(refused)


def test_a_file_never_built_before_is_built_and_counted_not_skipped(tree: SimpleNamespace) -> None:
    def edit(root: Path) -> None:
        (root / "src/c.c").write_text("int C(void) { return 3; } /* x */\n")
        (root / "src/d.c").write_text("int D(void) { return 5; }\n")

    result = _run(tree, _patch(tree, "new.diff", edit))
    assert result["compared"] == 2 and result["added"] == 2  # c.c in both versions; d.c in both, counted as added
    assert (tree.root / "build/us/src/d.c.o").is_file()


def test_a_change_reaching_no_unit_proves_nothing(tree: SimpleNamespace) -> None:
    def edit(root: Path) -> None:
        (root / "README.md").write_text("y\n")

    with pytest.raises(Refusal) as refused:
        _run(tree, _patch(tree, "doc.diff", edit))
    assert _keys(refused) == {"buildproof.empty"}


def test_a_dirty_tree_refuses(tree: SimpleNamespace) -> None:
    (tree.root / "src/c.c").write_text("changed\n")
    with pytest.raises(Refusal) as refused:
        _run(tree)
    assert _keys(refused) == {"buildproof.tree"}


def test_faults_tell_an_absent_unit_from_a_failure() -> None:
    quiet = SimpleNamespace(exit=2, stderr=b"make: *** No rule to make target 'build/de/x.o'.  Stop.\n")
    assert buildproof._faults(quiet, ["build/de/x.o"]) == ({"build/de/x.o"}, [])  # type: ignore[arg-type]
    needed = SimpleNamespace(exit=2, stderr=b"make: *** No rule to make target 'a.s', needed by 'build/de/x.o'.\n")
    assert buildproof._faults(needed, ["build/de/x.o"])[1]  # type: ignore[arg-type]
    silent = SimpleNamespace(exit=2, stderr=b"")
    assert buildproof._faults(silent, ["build/de/x.o"]) == (set(), ["make exit 2"])  # type: ignore[arg-type]
