from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abucache import __version__ as __version__
from abucache import log as log
from abucache.store import ContentCache as ContentCache

Runner = Callable[[str, Sequence[str], Path], Any]

@dataclass(frozen=True)
class Step:
    phase: str
    out: str
    argv: tuple[str, ...]

class ChainFailed(Exception):
    def __init__(self, step: Step, result: Any) -> None: ...

def identity(files: Sequence[Path], pins: Sequence[str] = ()) -> str: ...
def key(source: bytes, name: str, steps: Sequence[Step], tools: str) -> str: ...
def build(
    cache: ContentCache,
    run: Runner,
    steps: Sequence[Step],
    source: Path,
    tools: str,
    work: Path,
    after: Callable[[Step, Any], None] | None = None,
) -> tuple[bytes, list[Any]]: ...
