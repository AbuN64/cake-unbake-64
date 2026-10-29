"""`unbake lanecheck`: a hand input may touch only the files its lane owns.

`decomp cleanup slices` writes one exclusive file set per lane (`<names>/files/<LANE>.json`) and a GLOBALS set of the
files two or more modules read (`<names>/files/GLOBALS.json`). A lane with a file set is a hand lane: its diff may not
change a file another lane or GLOBALS owns, and each refusal names that owner (GLOBALS belongs to tool batches, which
carry renames across files). A lane with no file set is a tool batch and is not checked. A file in no set (a new file,
the Makefile, a config) is nobody's. No project, lane or file is special: the sets are the whole input."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from unbake.contracts import Finding, Json, Refusal

GLOBALS = "GLOBALS"
HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$", re.MULTILINE)


def touched(text: str) -> list[str]:
    """The files a diff changes: both sides of every `diff --git` section (a rename touches its old and new file)."""
    return sorted({name for pair in HEADER.findall(text) for name in pair})


def _set(names: Path, lane: str) -> list[str] | None:
    try:
        return [str(f) for f in json.loads((names / "files" / f"{lane}.json").read_text(encoding="utf-8"))["files"]]
    except (OSError, ValueError, KeyError):
        return None


def owners(names: Path) -> dict[str, str]:
    """file -> its lane (GLOBALS for a shared file), over every file set under NAMES."""
    out: dict[str, str] = {}
    for path in sorted((names / "files").glob("*.json")):
        lane = path.stem
        out.update({name: lane for name in _set(names, lane) or ()})
    return out


def check(names: Path, lane: str, diffs: Sequence[Path]) -> Json:
    """The files `diffs` touch; a Refusal naming the owner of every one outside LANE's set."""
    files = sorted({name for diff in diffs for name in touched(diff.read_text(encoding="utf-8", errors="replace"))})
    held = _set(names, lane)
    if held is None:
        return {"lane": lane, "hand": False, "files": files}
    owner = owners(names)
    outside = [name for name in files if owner.get(name, lane) != lane]
    if outside:
        raise Refusal(
            *(
                Finding(
                    "lane.outside",
                    f"{name} is owned by {owner[name]}, not by {lane}"
                    + (" (tool batches only)" if owner[name] == GLOBALS else ""),
                    path=name,
                    missing=(owner[name],),
                    action=(
                        "drop the file from the diff, or hand it to its owning lane; renames cross files by tool batch"
                    ),
                )
                for name in outside
            )
        )
    return {"lane": lane, "hand": True, "files": files}


def lane_command(params: Json) -> Json:
    """The command: no host or ROM is read, only the file sets and the diffs."""
    return check(Path(params["names"]), params["lane"], [Path(p) for p in params["diff"]])
