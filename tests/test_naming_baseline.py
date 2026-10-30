"""A reviewed naming budget shrinks, never masks a new occurrence or source file."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from unbake import api, naming_baseline, policy
from unbake.contracts import Refusal


def baseline(root, files):
    doc = {"schema": 1, "revision": "a" * 40, "files": naming_baseline.by_file(api.naming_findings(files))}
    path = root / naming_baseline.PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setattr(policy.shutil, "which", lambda _: "git")
    monkeypatch.setattr(policy.process, "scratch", lambda root: tmp_path)

    def run(name, argv, root, **kwargs):
        paths = sorted(str(p.relative_to(root)) for p in root.glob("src/*.c"))
        return SimpleNamespace(exit=0, stdout="\0".join(paths).encode() if argv[1] == "ls-files" else b"")

    monkeypatch.setattr(policy.process, "run", run)
    monkeypatch.setattr(policy, "_offset_audit", lambda *a: ([], []))
    monkeypatch.setattr(policy, "_removed_functions", lambda *a: 0)
    monkeypatch.setattr(policy, "_credit_drops", lambda *a: {})
    (tmp_path / "src").mkdir()
    return lambda: policy.audit_command({"project": str(tmp_path), "against": "HEAD"})


def test_audit_keeps_debt_shrinks_fixed_occurrences_and_refuses_reintroduction(tmp_path, audit):
    text = "int __private;\nint __private;\n"
    file = tmp_path / "src/a.c"
    file.write_text(text)
    path = baseline(tmp_path, {"src/a.c": text})
    report = audit()
    assert len(report["notes"]) == 2 and all(not f["blocking"] for f in report["notes"])
    file.write_text("\nint __private;\n")
    assert len(audit()["notes"]) == 1
    assert api.naming_debt(tmp_path)["src/a.c"][0]["lines"] == [2]
    reduced = path.read_bytes()
    file.write_text(text)
    with pytest.raises(Refusal) as error:
        audit()
    assert len(error.value.findings) == 1
    assert path.read_bytes() == reduced
    file.write_text("int portable;\n")
    assert audit()["naming_debt"] == {}
    assert json.loads(path.read_text())["files"] == {}
    file.write_text("int __private;\n")
    with pytest.raises(Refusal):
        audit()


@pytest.mark.parametrize("change", ["new-file", "new-reason", "asm-label"])
def test_new_findings_fail_without_mutating_reviewed_baseline(tmp_path, audit, change):
    original = "int __private;\n"
    file = tmp_path / "src/a.c"
    file.write_text(original)
    path = baseline(tmp_path, {"src/a.c": original})
    held = path.read_bytes()
    if change == "new-file":
        (tmp_path / "src/b.c").write_text(original)
    elif change == "new-reason":
        file.write_text(original + "int class;\n")
    else:
        file.write_text(original + 'extern int Function(void) __asm__("Other");\n')
    with pytest.raises(Refusal):
        audit()
    assert path.read_bytes() == held


def test_failed_other_gate_cannot_shrink_baseline(tmp_path, audit, monkeypatch):
    path = baseline(tmp_path, {"src/a.c": "int __private;\n"})
    (tmp_path / "src/a.c").write_text("int portable;\n")
    original = path.read_bytes()
    monkeypatch.setattr(policy, "_credit_drops", lambda *a: {"us": 1})
    with pytest.raises(Refusal):
        audit()
    assert path.read_bytes() == original


def test_missing_baseline_is_strict_and_bad_config_is_not_ignored(tmp_path, audit):
    (tmp_path / "src/a.c").write_text("int __private;\n")
    with pytest.raises(Refusal):
        audit()
    path = baseline(tmp_path, {})
    path.write_text('{"schema": 1, "files": {}}')
    with pytest.raises(Refusal) as error:
        audit()
    assert error.value.findings[0].key == "config.schema"


def test_candidate_admission_and_fresh_share_reviewed_budget(tmp_path, monkeypatch):
    original = "int __private;\n"
    baseline(tmp_path, {"src/a.c": original})
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(root=tmp_path, libraries=(), audit=None)),
        peek=lambda path: original.encode(),
    )
    monkeypatch.setattr(policy, "_module", lambda *a: None)
    monkeypatch.setattr(policy, "_findings", lambda snap, path, text, *args: api.naming_findings({path: text}))
    found = policy.evaluate(snapshot, "src/a.c", original + "int class;\n", None, False, {"symbol-name"})
    assert len(found) == 2 and sum(f.blocking for f in found) == 1
    assert len(policy.fresh(snapshot, "src/a.c", found, False)) == 1
    # A new naming failure cannot be grandfathered just because HEAD already has it.
    snapshot.peek = lambda path: b"int class;\n"
    bad = next(f for f in found if f.blocking)
    assert policy.fresh(snapshot, "src/a.c", [bad], False) == (bad,)


def test_multiset_is_per_file_and_ignores_line_shifts(tmp_path):
    found = api.naming_findings({"src/a.c": "int class;\n"})
    doc = {"files": naming_baseline.by_file(found)}
    shifted = [replace(found[0], line=90)]
    assert naming_baseline.split(shifted, doc)[0] == ()
    assert len(naming_baseline.split(shifted * 2, doc)[0]) == 1
    assert len(naming_baseline.split([replace(found[0], path="src/b.c")], doc)[0]) == 1


@pytest.mark.parametrize("name", ["__osPopThread", "__OSInode", "_G_CC_BLENDPE", "_GBI_V3"])
def test_sdk_implementation_names_are_source_only_exemptions(name):
    assert api.naming_findings({"src/a.c": f"extern int {name};\n"}) == ()
    assert api.name_issues(name)


def test_sdk_exemption_does_not_cover_game_names_or_alias_hacks():
    for text in (
        "int CVector3__Method;",
        "int __private;",
        "int _Global;",
        "int __osfake;",
        'extern int __osPopThread(void) __asm__("Other");',
        "int __os" + "A" * 248 + ";",
    ):
        assert api.naming_findings({"src/a.c": text})
