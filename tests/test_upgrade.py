"""upgrade: the tool replaces itself only while nothing runs from its environment."""

import sys
from types import SimpleNamespace

import pytest

from unbake import repo
from unbake.contracts import Refusal


def _procs(monkeypatch, lines):
    procs = [SimpleNamespace(info={"pid": 1000 + n, "cmdline": line}) for n, line in enumerate(lines)]
    monkeypatch.setattr(repo.psutil, "process_iter", lambda attrs: procs)


def test_upgrade_refuses_while_any_command_runs_from_the_environment(monkeypatch):
    # 2026-10-15 18:36 a venv swap under a running setup (cc6dec6) cost that lane its setup: the output would mix
    # two tool versions. Any command counts, not only land and submit.
    bin_ = f"{sys.prefix}/bin"
    _procs(monkeypatch, [[f"{bin_}/python", f"{bin_}/unbake", "setup"], ["/usr/bin/vim", "notes"]])
    monkeypatch.setattr(repo.process, "run", lambda *a, **k: pytest.fail("pip ran under a live command"))
    with pytest.raises(Refusal) as refused:
        repo.upgrade({"ref": "abc123"})
    (finding,) = refused.value.findings
    assert finding.key == "upgrade.busy" and len(finding.missing) == 1 and "unbake setup" in finding.missing[0]


def test_upgrade_installs_the_ref_from_its_own_source_when_nothing_runs(monkeypatch):
    _procs(monkeypatch, [["/other/venv/bin/python", "/other/venv/bin/unbake", "land"], None])
    monkeypatch.setattr(repo, "_source", lambda: "https://example.invalid/tool")
    ran = []
    monkeypatch.setattr(
        repo.process,
        "run",
        lambda name, argv, cwd, **k: ran.append(argv) or SimpleNamespace(exit=0, signal=None, stderr=b""),
    )
    assert repo.upgrade({"ref": "abc123"}) == {
        "ref": "abc123",
        "prefix": sys.prefix,
        "source": "https://example.invalid/tool",
    }
    assert ran[0][0] == sys.executable and ran[0][-1] == "git+https://example.invalid/tool@abc123"


def test_upgrade_ignores_its_own_process(monkeypatch):
    bin_ = f"{sys.prefix}/bin"
    procs = [SimpleNamespace(info={"pid": repo.os.getpid(), "cmdline": [f"{bin_}/unbake", "upgrade", "x"]})]
    monkeypatch.setattr(repo.psutil, "process_iter", lambda attrs: procs)
    assert repo._users(repo.Path(sys.prefix)) == []
