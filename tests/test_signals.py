"""A SIGTERM unwinds like an exception, so a trial landing's files are put back (10-24 score --alloc under timeout)."""

import signal
import subprocess
import sys
import textwrap


def test_sigterm_runs_the_restoring_finally_block(tmp_path):
    marker = tmp_path / "unit.c"
    marker.write_text("kept")
    code = textwrap.dedent(
        f"""
        import os, sys, time
        from pathlib import Path
        from unbake import cli
        cli.unwind_on_signals()
        unit = Path({str(marker)!r})
        old = unit.read_bytes()
        try:
            unit.unlink()
            print("ready", flush=True)
            time.sleep(30)
        finally:
            unit.write_bytes(old)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(10) == 143
    assert marker.read_text() == "kept"
