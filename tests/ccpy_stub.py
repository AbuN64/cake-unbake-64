"""CCPY for tests that make fake objects: the real helper, its trim only logs ('trimmed' in the working folder)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unbake" / "resources"))
import cc_cache


def logged(path: str, size: str) -> None:
    with Path("trimmed").open("a") as log:
        log.write("x\n")


cc_cache.trim_text = logged
sys.exit(cc_cache.main(sys.argv))
