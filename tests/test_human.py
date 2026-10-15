"""Plain findings, exact summaries, and terminal-only progress and labels."""

import io
import os
from dataclasses import replace

import pytest

from unbake import effort, human
from unbake.contracts import Finding, Origin, StageRecord


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    monkeypatch.setattr(effort, "_listeners", [])
    monkeypatch.setattr(effort, "closed", lambda: ())
    monkeypatch.setattr(human, "_stream", None)
    monkeypatch.setattr(human, "_tty", False)
    monkeypatch.setattr(human, "_live", False)
    monkeypatch.setattr(human.config, "sentence", lambda key: "plain sentence")


@pytest.mark.parametrize("blocking,prefix", [(True, "Refused"), (False, "Debt")])
def test_finding_nested_english(blocking, prefix):
    finding = Finding(
        "effort.log",
        "the reason",
        "src/file.c",
        7,
        "member",
        ("a", "b"),
        ("bytes", "symbols"),
        blocking=blocking,
        action="unbake compare",
    )
    assert human.finding(finding) == (
        f"{prefix}: plain sentence.\n  because the reason\n  in src/file.c:7\n"
        "  item member\n  versions a, b\n  missing bytes\n  missing symbols\n  next: unbake compare"
    )


@pytest.mark.parametrize("line", [0, -1, 2])
def test_finding_optional_fields(line):
    suffix = ":2" if line == 2 else ""
    assert human.finding(Finding("effort.log", "", "file", line)) == f"Refused: plain sentence.\n  in file{suffix}"
    assert human.finding(Finding("effort.log", "")) == "Refused: plain sentence."


@pytest.mark.parametrize("tty", [False, True])
def test_status_line_only_on_tty(monkeypatch, tty):
    stream = io.StringIO()
    monkeypatch.setattr(human.shutil, "get_terminal_size", lambda: os.terminal_size((12, 20)))
    human.attach(stream, tty)
    effort._notify("open", {"path": ["check", "longstage"]})
    effort.progress("units", 2, 4)
    done = {"status": "ok", "cores": 0, "jobs": 0}
    effort._notify("close", {"path": ["check", "longstage"], "wall": 1, **done})
    effort._notify("close", {"path": ["check"], "wall": 2, **done})
    done = "check > longstage 1.0s 0.00 cores 0 jobs\ncheck 2.0s 0.00 cores 0 jobs\n"
    live = "\r\x1b[Kcheck > lon\r\x1b[Kunits 2/4\r\x1b[K"
    assert stream.getvalue() == "check > longstage started\n" + (live if tty else "") + done
    assert not human._live


def _records():
    keys = ("resources.memory_parent_bytes", "resources.memory_worker_bytes")
    root = StageRecord(
        "inv",
        "root",
        None,
        ("check",),
        "command",
        "ok",
        "parallel",
        0,
        10,
        2,
        6,
        12,
        2,
        4,
        4,
        2,
        3,
        2_000_000,
        3_000_000,
        {key: Origin(key, "/host.toml", "0" * 64) for key in keys},
        dict(zip(keys, (100_000_000, 200_000_000), strict=True)),
        {"view": (1, 2), "object": (3, 4)},
    )
    stage = replace(root, span="stage", parent="root", path=("check", "work"), kind="stage", wall_seconds=5)
    own = replace(
        stage,
        span="own",
        path=("check", "(own)"),
        kind="own",
        wall_seconds=2,
        findings=(Finding("budget.single_core", "too slow", blocking=False),),
    )
    return [own, stage, root]


def test_summary_lines():
    assert human.summary(_records()) == [
        "took 10.00s, 2.00 cores (parent 2.00s, workers 6.00s, native 12.00s cpu)",
        "memory: parent peak 2.0 MB of 100.0 MB (/host.toml), workers peak 3.0 MB of 200.0 MB (/host.toml)",
        "cache: object 3 warm / 4 cold",
        "cache: view 1 warm / 2 cold",
        "slowest:",
        "  check  10.00s  2.00 cores  parallel  items 4 jobs 4 workers 2/3",
        "  check > work  5.00s  2.00 cores  parallel  items 4 jobs 4 workers 2/3",
        "  check > (own)  2.00s  2.00 cores  parallel  items 4 jobs 4 workers 2/3",
        "slow: plain sentence too slow",
    ]


def test_summary_last_root_and_ten_slowest():
    root = _records()[-1]
    records = [replace(root, span=str(i), kind="stage", wall_seconds=i, path=(str(i),)) for i in range(12)]
    records += [root, replace(root, wall_seconds=20)]
    lines = human.summary(records)
    assert lines[0].startswith("took 20.00s")
    assert len(lines[lines.index("slowest:") + 1 :]) == 10
    assert human.summary([]) == []
    assert human.summary(records[:12]) == []


def test_finish_clears_and_orders_findings(monkeypatch):
    stream = io.StringIO()
    human.attach(stream, True)
    effort.progress("units", 1, 2)
    monkeypatch.setattr(effort, "closed", _records)
    human.finish("check", None, [Finding("effort.log", "debt", blocking=False), Finding("effort.log", "block")])
    text = stream.getvalue()
    assert text.startswith(
        "\r\x1b[Kunits 1/2\r\x1b[KRefused: plain sentence.\n  because block\nDebt: plain sentence.\n  because debt\n"
    )
    assert "took 10.00s" in text
    assert not human._live


def test_written_files_print_the_exact_add_and_one_line_commit() -> None:
    landed = {
        "running": False,
        "refused": [],
        "landed": [
            {
                "id": "i",
                "member": "f",
                "operation": "publish",
                "files": ["src/a b.c", "layout.toml"],
                "message": "publish f (it's)",
            }
        ],
    }
    assert human._written("land", landed) == [
        "Written, not committed. To commit:",
        "  git add -A -- 'src/a b.c' layout.toml",
        "  git commit -m 'publish f (it'\"'\"'s)'",
    ]
    assert human._written("move", {"files": ["x"], "message": "move: a -> b"})[-1] == "  git commit -m 'move: a -> b'"
    assert human._written("check", {"files": ["x"]}) == [] and human._written("land", None) == []


@pytest.mark.parametrize(
    "path,expected,printed",
    [
        (["a"], None, True),  # shallow and never seen: always announced
        (["a", "b"], None, True),
        (["a", "b", "c"], None, False),  # depth three is announced only when the last run was slow
        (["a", "b", "c"], 30.0, True),
        (["a", "b"], 3.0, False),
        (["a", "b", "c", "d"], 99.0, False),  # deeper stages print nothing
        (["job", "native.objects"], None, False),  # a worker's per-item job prints nothing
    ],
)
def test_start_lines(path, expected, printed):
    stream = io.StringIO()
    human.attach(stream, False)
    effort._notify("open", {"path": path, "expected": expected})
    assert bool(stream.getvalue()) is printed
    assert "\x1b" not in stream.getvalue()


def test_a_said_line_reaches_the_person_as_one_line():
    stream = io.StringIO()
    human.attach(stream, False)
    effort.say("unbake: waiting for pids 7 to finish")
    assert stream.getvalue() == "unbake: waiting for pids 7 to finish\n"
