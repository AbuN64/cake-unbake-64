from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from abucache import log as log
from abucache.compile import Runner as Runner

class Owned(Protocol):
    @property
    def section(self) -> str: ...
    @property
    def vram(self) -> int: ...
    @property
    def rom_start(self) -> int: ...
    @property
    def size(self) -> int: ...

class LinkFailed(Exception):
    reason: str
    result: Any
    def __init__(self, reason: str, result: Any = None) -> None: ...

@dataclass(frozen=True)
class Linked:
    result: Any
    outputs: dict[str, Path]

def section_file(section: str) -> str: ...
def link_script(
    sections: Sequence[Owned], symbols: str, table: Mapping[str, Mapping[str, Any]], *, trim: bool = False
) -> str: ...
def link_commands(
    tools: Mapping[str, str], obj: Path, sections: Sequence[Owned], work: Path, trim: bool, *, score: bool = False
) -> tuple[tuple[str, ...], ...]: ...
def object_symbols(obj: Path) -> set[bytes]: ...
def needed_symbols(obj: Path, symbols: bytes) -> bytes: ...
def link(
    run: Runner,
    tools: Mapping[str, str],
    obj: Path,
    sections: Sequence[Owned],
    symbols: Path,
    work: Path,
    trim: bool,
    table: Mapping[str, Mapping[str, Any]],
    *,
    score: bool = False,
) -> Linked: ...
