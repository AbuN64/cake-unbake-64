"""Source-wide rules: forbidden imports, dispatch literals, finding keys, line budgets, contract-only calls."""

from __future__ import annotations

import ast
import re
import tomllib
import warnings
from pathlib import Path

import pytest

SOFT_CAP = 6500
SRC = Path(__file__).resolve().parents[1] / "src" / "unbake"
FILES = sorted(p for p in SRC.glob("*.py") if p.name != "contracts.py")
CONTRACT_TEXT = (SRC / "resources" / "CONTRACTS.md").read_text(encoding="utf-8")
BUDGETS = {name: int(budget) for name, budget in re.findall(r"^- (\w+): (\d+)$", CONTRACT_TEXT, re.M)}
API = {
    module: " ".join(re.findall(r"`(\w+)(?:\(|:)", row))
    for module, row in re.findall(r"^\| (\w+) \| (.+) \|$", CONTRACT_TEXT, re.M)
    if module in BUDGETS
}
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
CONTRACTS = "unbake.contracts"
PROCESS_OWNERS = {"process.py", "pool.py", "effort.py"}
FORBIDDEN = ("subprocess", "multiprocessing", "concurrent.futures", "threading")
DISPATCH = {"gcc", "ido", "c", "asm", "data", "resource"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _docstrings(tree: ast.Module) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOCUMENTED) and ast.get_docstring(node):
            found.add(id(node.body[0].value))
    return found


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    return [node.module or ""] if isinstance(node, ast.ImportFrom) else []


@pytest.mark.parametrize("path", [p for p in FILES if p.name not in PROCESS_OWNERS], ids=lambda p: p.name)
def test_no_process_machinery(path: Path) -> None:
    tree = _tree(path)
    for node in ast.walk(tree):
        names = _imported(node)
        for name in names:
            banned = any(name == f or name.startswith(f + ".") for f in FORBIDDEN)
            assert not banned, f"{path.name}:{node.lineno} imports {name}"
        if isinstance(node, ast.Attribute) and node.attr == "fork" and getattr(node.value, "id", "") == "os":
            raise AssertionError(f"{path.name}:{node.lineno} uses os.fork")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_dispatch_literals(path: Path) -> None:
    tree = _tree(path)
    skip = _docstrings(tree)
    # Dispatch means branching on a kind/toolchain name: comparisons and match cases. Paths and keys are data.
    for node in ast.walk(tree):
        operands = [node.left, *node.comparators] if isinstance(node, ast.Compare) else []
        operands += [node.value] if isinstance(node, ast.MatchValue) else []
        for operand in operands:
            for leaf in ast.walk(operand):
                if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str) and id(leaf) not in skip:
                    assert leaf.value not in DISPATCH, f"{path.name}:{leaf.lineno} dispatches on {leaf.value!r}"


def test_finding_keys_exist() -> None:
    keys = set(tomllib.loads((SRC / "resources/data/keys.toml").read_text(encoding="utf-8"))["key"])
    for path in FILES:
        for node in ast.walk(_tree(path)):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Finding"):
                continue
            first = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "key"), None)
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                assert first.value in keys, f"{path.name}:{node.lineno} Finding key {first.value!r} is not in keys.toml"


def _count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def _lines(names: list[str]) -> int:
    return sum(_count(SRC / f"{n}.py") for n in names)


def test_line_budgets() -> None:
    # Owner 2026-10-14: the line cap is soft; module allocations are planning numbers.
    unknown = {p.stem for p in FILES} - set(BUDGETS) - {"__init__", "__main__"}
    assert not unknown, f"modules without a budget: {sorted(unknown)}"
    total = _lines([p.stem for p in FILES]) + _count(SRC / "contracts.py")
    if total > SOFT_CAP:
        warnings.warn(f"source total {total} is over the soft cap of {SOFT_CAP:,}", stacklevel=1)  # non-blocking


def _aliases(tree: ast.Module) -> dict[str, str]:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "unbake":
            out.update({a.asname or a.name: a.name for a in node.names if a.name in API})
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_only_contract_names_cross_modules(path: Path) -> None:
    tree = _tree(path)
    aliases = _aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            module, attr = aliases[node.value.id], node.attr
            ok = attr.startswith("__") or attr in API[module].split()
            assert ok, f"{path.name}:{node.lineno} uses {module}.{attr}, not in CONTRACTS.md"
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("unbake.") and node.module != CONTRACTS:
            module = node.module.split(".", 1)[1]
            if module in API:
                bad = [a.name for a in node.names if a.name not in API[module].split()]
                assert not bad, f"{path.name}:{node.lineno} imports {bad} from {module}, not in CONTRACTS.md"


@pytest.mark.parametrize("module", sorted(set(API) - {"cli"}))
def test_public_functions_are_contract_rows(module: str) -> None:
    path = SRC / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module}.py not written yet")
    functions = (n for n in _tree(path).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    public = {n.name for n in functions if not n.name.startswith("_")}
    extra = sorted(public - set(API[module].split()))
    assert not extra, f"{module} has public functions outside its row: {extra}"


REMOVED = (
    "store.claim",
    "ledger_append",
    "store.ledger(",
    "features.toml",
    "build.init",
    "build.setup",
    "publish.publish",
    "from unbake import queue",
    "from unbake import search",
    '"build" / "stages.jsonl"',
)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_removed_names_absent(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    found = [name for name in REMOVED if name in text]
    assert not found, f"{path.name} still contains removed names {found}"


def test_only_human_writes_stderr() -> None:
    allowed = {"human.py", "cli.py", "effort.py"}
    for path in FILES:
        if path.name not in allowed:
            assert "sys.stderr" not in path.read_text(encoding="utf-8"), f"{path.name} writes to sys.stderr"


def test_result_schemas_exist() -> None:
    commands = tomllib.loads((SRC / "resources/data/commands.toml").read_text(encoding="utf-8"))["command"]
    wanted = {f"result.{name}.schema.json" for name in commands}
    present = {p.name for p in (SRC / "resources/schemas").glob("result.*.schema.json")}
    assert wanted <= present, f"missing result schemas: {sorted(wanted - present)}"


DELETED = (
    "layout",
    "ownership",
    "infer",
    "move",
    "moved",
    "headers",
    "types",
    "declarations",
    "retype",
    "recovery",
    "libasm",
    "families",
)
PROJECT_NAMES = ("RageWars", "BattleTanx", "Turok")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_deleted_modules_are_not_imported(path: Path) -> None:
    for node in ast.walk(_tree(path)):
        for name in _imported(node):
            parts = name.split(".")
            assert not (parts[0] == "unbake" and len(parts) > 1 and parts[1] in DELETED), f"{path.name} imports {name}"
        if isinstance(node, ast.ImportFrom) and node.module == "unbake":
            bad = [a.name for a in node.names if a.name in DELETED]
            assert not bad, f"{path.name}:{node.lineno} imports deleted modules {bad}"


def test_deleted_modules_are_gone() -> None:
    assert not [m for m in DELETED if (SRC / f"{m}.py").exists() or (SRC / m).exists()]


def test_no_project_name_in_the_tool_source() -> None:
    found = [
        f"{p.relative_to(SRC)}: {name}"
        for p in sorted(SRC.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".toml", ".in", ".json", ".md"}
        for name in PROJECT_NAMES
        if name.lower() in p.read_text(encoding="utf-8", errors="replace").lower()
    ]
    assert not found, f"project names in src/: {found}"
