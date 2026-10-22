"""treelock: readers share the tree, a writer holds it alone, and a waiting reader says so instead of failing."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from unbake import treelock

HOLD = (
    "import sys,time;from pathlib import Path;from unbake import treelock\n"
    "with treelock.hold(Path(sys.argv[1]), exclusive=sys.argv[2]=='x'):\n"
    "    print('held',flush=True);time.sleep(float(sys.argv[3]))\n"
)


def _other(root: Path, mode: str, seconds: float) -> subprocess.Popen[str]:
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", HOLD, str(root), mode, str(seconds)],
        stdout=subprocess.PIPE,
        text=True,
        env={"PYTHONPATH": str(Path(treelock.__file__).parents[1])},
    )
    assert child.stdout and child.stdout.readline().strip() == "held"
    return child


def test_readers_share_and_a_reader_waits_out_a_writer(tmp_path: Path) -> None:
    said: list[str] = []
    other = _other(tmp_path, "s", 0.3)
    with treelock.hold(tmp_path) as waited:  # a shared hold beside a shared one: no wait
        assert waited < 0.2
    other.wait()
    writer = _other(tmp_path, "x", 1.0)
    began = time.monotonic()
    with treelock.hold(tmp_path, who="try f", say=said.append) as waited:
        assert waited >= 0.5 and time.monotonic() - began >= 0.5
    writer.wait()
    assert said and said[0].startswith("waiting for ")


def test_a_holder_marks_its_children_so_they_run_inside_the_hold(tmp_path: Path) -> None:
    with treelock.hold(tmp_path, exclusive=True), treelock.hold(tmp_path) as waited:  # would deadlock without the mark
        assert waited == 0.0
    assert treelock.ENV not in __import__("os").environ


LOOP = (
    "import sys,time;from pathlib import Path;from unbake import treelock\n"
    "end=time.monotonic()+float(sys.argv[2])\n"
    "while time.monotonic()<end:\n"
    "    with treelock.hold(Path(sys.argv[1])):\n"
    "        time.sleep(0.4)\n"
)


def test_a_stream_of_readers_never_starves_a_writer_and_the_writer_bumps_the_epoch(tmp_path: Path) -> None:
    env = {"PYTHONPATH": str(Path(treelock.__file__).parents[1])}
    readers = [
        subprocess.Popen([sys.executable, "-I", "-c", LOOP, str(tmp_path), "6"], env=env, stderr=subprocess.DEVNULL)
        for _ in range(4)
    ]  # back to back, overlapping: without a writer's announcement the lock is never free
    time.sleep(1.0)
    before = treelock.epoch(tmp_path)
    with treelock.hold(tmp_path, exclusive=True, who="ship", say=lambda text: None) as waited:
        assert waited < 1.5  # within about one reader's hold, not when the stream ends
    assert treelock.epoch(tmp_path) == before + 1
    for reader in readers:
        reader.wait()


def test_a_reader_behind_a_writer_names_it(tmp_path: Path) -> None:
    said: list[str] = []
    writer = _other(tmp_path, "x", 1.0)
    with treelock.hold(tmp_path, say=said.append):
        pass
    writer.wait()
    assert said and "waiting for" in said[0]
