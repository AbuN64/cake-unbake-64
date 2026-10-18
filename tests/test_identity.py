"""code_identity: a key that guards a producer moves with the code the producer reaches and with nothing else."""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

from unbake import contracts, native, policy
from unbake.contracts import code_identity


def _module(source: str) -> dict:
    scope = {"__name__": "fixturepkg.mod"}
    exec(compile(source, "mod.py", "exec"), scope)  # a module of its own package, rebuilt per edit
    return scope


_BASE = textwrap.dedent(
    """
    import re
    LIMIT = 3
    PATTERN = re.compile("a+")
    CACHE = {}

    def helper(x):
        return x + LIMIT

    def unrelated(x):
        return x * 2

    def produce(x):
        return [helper(i) for i in range(x)]

    def lambdas(x):
        return sorted(x, key=lambda v: -v)
"""
)


def _write(directory: Path, source: str) -> str:
    path = directory / f"mod{len(source)}.py"
    path.write_text(source)
    contracts._assigned.cache_clear()
    return str(path)


def _identity(source: str, name: str = "produce") -> str:
    scope = _module(source)
    code_identity.cache_clear()
    return code_identity(scope[name])


def test_an_edit_of_an_unrelated_function_leaves_the_identity_equal():
    assert _identity(_BASE) == _identity(_BASE.replace("return x * 2", "return x * 3 + 1"))


def test_line_shifts_alone_leave_the_identity_equal():
    assert _identity(_BASE) == _identity("# a comment\n\n\n" + _BASE.replace("def helper", "\n\ndef helper"))


def test_an_edit_of_a_callee_a_constant_or_a_nested_lambda_changes_it():
    base = _identity(_BASE)
    assert _identity(_BASE.replace("return x + LIMIT", "return x - LIMIT")) != base
    assert _identity(_BASE.replace("LIMIT = 3", "LIMIT = 4")) != base
    assert _identity(_BASE.replace("range(x)", "range(x + 1)")) != base
    assert _identity(_BASE, "lambdas") != _identity(_BASE.replace("-v", "v"), "lambdas")


def test_mutable_module_state_is_named_by_its_type_only():
    scope = _module(_BASE + "\ndef reads():\n    return CACHE\n")
    code_identity.cache_clear()
    before = code_identity(scope["reads"])
    scope["CACHE"]["filled"] = [1, 2, 3]
    code_identity.cache_clear()
    assert code_identity(scope["reads"]) == before


def test_a_module_root_covers_code_reached_only_by_name():
    scope = _module(_BASE)
    scope["__name__"] = "fixturepkg.mod"
    module = type(contracts)("fixturepkg.mod")
    module.__dict__.update(scope)
    code_identity.cache_clear()
    first = code_identity(module)
    scope["unrelated"] = _module("def unrelated(x):\n    return 0\n")["unrelated"]
    module.__dict__.update(scope)
    code_identity.cache_clear()
    assert code_identity(module) != first


def _moved(module, name: str, roots) -> bool:
    code_identity.cache_clear()
    before = code_identity(*roots)
    original = getattr(module, name)

    def other(*args, **kwargs):
        return 12345

    other.__module__ = module.__name__
    setattr(module, name, other)
    try:
        code_identity.cache_clear()
        return code_identity(*roots) != before
    finally:
        setattr(module, name, original)
        code_identity.cache_clear()


def test_the_census_identity_ignores_an_unrelated_policy_function_and_follows_a_reached_one():
    roots = (policy.evaluate, policy._view_uses, policy._stand_in_body)
    assert not _moved(policy, "_census_warnings", roots)
    assert not _moved(policy, "unclean", roots)
    assert _moved(policy, "_findings", roots)


def test_the_stamp_identity_ignores_unrelated_native_functions_and_follows_a_reached_one():
    roots = (native.measure, native.objects, native.warnings, native.recorded, native.dumps)
    assert not _moved(native, "stamp", roots)  # the stamp reads the measure's code, not the other way round
    assert _moved(native, "_anchors", roots)


def test_a_table_is_named_by_the_expression_that_built_it_not_by_what_it_holds(tmp_path):
    table = "\nTABLE = {'a': 1}\n\ndef reads():\n    return TABLE\n"
    scope = _module(table)
    sys.modules["fixturepkg.mod"] = types.SimpleNamespace(__file__=_write(tmp_path, table))
    try:
        code_identity.cache_clear()
        before = code_identity(scope["reads"])
        scope["TABLE"]["filled"] = 2  # a memo growing at run time
        code_identity.cache_clear()
        assert code_identity(scope["reads"]) == before
        other = _module(table.replace("'a': 1", "'a': 2"))
        sys.modules["fixturepkg.mod"] = types.SimpleNamespace(
            __file__=_write(tmp_path, table.replace("'a': 1", "'a': 2"))
        )
        code_identity.cache_clear()
        assert code_identity(other["reads"]) != before
    finally:
        del sys.modules["fixturepkg.mod"]
        contracts._assigned.cache_clear()


def _package(tmp_path: Path, sources: dict[str, str]) -> dict[str, types.ModuleType]:
    """Real modules of a throwaway package, each with a file on disk, registered in sys.modules in the given order."""
    made = {}
    for name, source in sources.items():
        module = types.ModuleType(f"fixturepkg.{name}")
        module.__file__ = str(tmp_path / f"{name}{len(source)}.py")
        Path(module.__file__).write_text(source)
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, "exec"), vars(module))
        made[name] = module
    contracts._assigned.cache_clear()
    code_identity.cache_clear()
    return made


def _drop_package() -> None:
    for name in [n for n in sys.modules if n.startswith("fixturepkg")]:
        del sys.modules[name]
    contracts._assigned.cache_clear()
    code_identity.cache_clear()


def test_an_imported_table_is_named_by_the_expression_in_the_module_that_built_it(tmp_path):
    reader = "from fixturepkg.tables import TABLE\n\ndef reads():\n    return TABLE['a']\n"
    try:
        before = code_identity(_package(tmp_path, {"tables": "TABLE = {'a': 1}\n", "reader": reader})["reader"].reads)
        _drop_package()
        after = code_identity(_package(tmp_path, {"tables": "TABLE = {'a': 2}\n", "reader": reader})["reader"].reads)
        assert after != before
    finally:
        _drop_package()


def test_a_table_built_from_another_global_moves_when_that_global_is_edited(tmp_path):
    source = "_ABI = ('$a0', '$a1')\nHARD = {n: i for i, n in enumerate(_ABI)}\n\ndef reads():\n    return HARD\n"
    try:
        before = code_identity(_package(tmp_path, {"alloc": source})["alloc"].reads)
        _drop_package()
        edited = source.replace("'$a1'", "'$a2'")
        assert code_identity(_package(tmp_path, {"alloc": edited})["alloc"].reads) != before
        _drop_package()
        assert code_identity(_package(tmp_path, {"alloc": source})["alloc"].reads) == before
    finally:
        _drop_package()


_COUNTER = "_N = 0\n\ndef bump():\n    global _N\n    _N = 5\n\ndef reads():\n    return _N\n"


def test_a_global_a_run_reassigns_leaves_the_identity_equal(tmp_path):
    try:
        mod = _package(tmp_path, {"mod": _COUNTER})["mod"]
        before = code_identity(mod.reads)
        mod.bump()
        code_identity.cache_clear()
        assert mod.reads() == 5 and code_identity(mod.reads) == before
    finally:
        _drop_package()


def test_editing_the_expression_of_a_reassigned_global_moves_the_identity(tmp_path):
    try:
        first = code_identity(_package(tmp_path, {"mod": _COUNTER})["mod"].reads)
        _drop_package()
        second = code_identity(_package(tmp_path, {"mod": _COUNTER.replace("_N = 0", "_N = 1")})["mod"].reads)
        assert first != second
    finally:
        _drop_package()


def test_the_identity_of_a_producer_is_equal_in_every_process():
    import subprocess

    program = (
        "from unbake import pool, rescore\nfrom unbake.contracts import code_identity\n"
        "print(code_identity(pool._keys), code_identity(rescore.scores))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", program],
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": ":".join(sys.path)},
            capture_output=True,
            check=True,
            text=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(runs) == 1  # a hash seed or a set order must never move a cache key


def test_code_of_a_plumbing_module_is_named_and_never_followed(monkeypatch):
    def identity(plumbing: bool, body: str) -> str:
        helper = types.ModuleType("fixturepkg.helper")
        helper.__plumbing__ = plumbing
        exec(compile(f"def schedule(x):\n    return {body}\n", "helper.py", "exec"), vars(helper))
        helper.schedule.__module__ = "fixturepkg.helper"
        monkeypatch.setitem(sys.modules, "fixturepkg.helper", helper)
        scope = _module("def produce(x):\n    return schedule(x) + 1\n")
        scope["schedule"] = helper.schedule
        code_identity.cache_clear()
        return code_identity(scope["produce"])

    assert identity(True, "x") == identity(True, "[x, x]")  # a scheduler edit leaves stored results warm
    assert identity(False, "x") != identity(False, "[x, x]")
    assert identity(True, "x") != identity(False, "x")
