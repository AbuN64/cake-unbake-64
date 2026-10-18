"""recipes: flag precedence, paired groups, refusals, proposals and digests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from unbake import config as configuration
from unbake import recipes
from unbake.contracts import Config, Refusal, UnitSpec, digest

PAIRED_ROW = {
    "cflags": ["-O2", "-G", "0"],
    "supported_options": ["-O1", "-O2", "-O3", "-mgp32", "-G"],
    "paired": ["-G"],
}


def _config(cfg: Config, **flags: list[str]) -> Config:
    """cfg with the project's cflags, cppflags or asflags replaced."""
    project = SimpleNamespace(**{**vars(cfg.project), **{k: tuple(v) for k, v in flags.items()}})
    return replace(cfg, project=project)  # type: ignore[arg-type]


def _unit(path: str = "src/a.c", toolchain: str = "gcc-test", **options: list[str]) -> UnitSpec:
    return UnitSpec(path, "c", ("f",), toolchain, options)


@pytest.fixture
def paired(toolchains: dict[str, Any]) -> str:
    toolchains["toolchain"]["gcc-paired"] = {**toolchains["toolchain"]["gcc-test"], **PAIRED_ROW}
    return "gcc-paired"


def test_precedence_order(cfg: Config, toolchains: dict[str, Any]) -> None:
    row = toolchains["toolchain"]["gcc-test"]
    toolchains["toolchain"]["gcc-test"] = {**row, "supported_options": ["-O1", "-O2", "-Wall", "-g"]}
    cfg = _config(cfg, cflags=["-P1", "-P2"])
    unit = _unit(add=["-V1", "-V2"], omit=["-P1"])
    recipe = recipes.resolve(cfg, unit, {"omit": ["-V1", "-O2"], "add": ["-Wall", "-g"]})
    assert recipe.cflags == ("-P2", "-V2", "-Wall", "-g")
    assert recipe.toolchain == "gcc-test"


def test_asflags_and_cppflags_from_build(cfg: Config) -> None:
    cfg = _config(cfg, cppflags=["-DX"], asflags=["-a"])
    recipe = recipes.resolve(cfg, _unit(), {})
    assert recipe.cppflags == ("-DX",)
    assert recipe.asflags == (*configuration.load_resource("rom.toml")["asflags"], "-a")
    assert recipe.digest == digest((recipe.toolchain, recipe.cppflags, recipe.cflags, recipe.asflags))


def test_toolchain_override(cfg: Config) -> None:
    recipe = recipes.resolve(cfg, _unit(), {"toolchain": "ido-7.1"})
    assert recipe.toolchain == "ido-7.1"
    assert "-non_shared" in recipe.cflags


def test_paired_omit(cfg: Config, paired: str) -> None:
    recipe = recipes.resolve(cfg, _unit(toolchain=paired), {"omit": ["-G"]})
    assert recipe.cflags == ("-O2",)
    recipe = recipes.resolve(cfg, _unit(toolchain=paired), {"omit": ["-G 0"]})
    assert recipe.cflags == ("-O2",)


def test_paired_add(cfg: Config, paired: str) -> None:
    recipe = recipes.resolve(cfg, _unit(toolchain=paired), {"omit": ["-G"], "add": ["-G 8"]})
    assert recipe.cflags == ("-O2", "-G", "8")


@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        ({"bogus": 1}, "recipe.option"),
        ({"omit": ["-O3"]}, "recipe.option"),
        ({"add": ["-nonsense"]}, "recipe.option"),
        ({"toolchain": "nope"}, "adapter.unknown"),
    ],
)
def test_unknown_option_refused(cfg: Config, overrides: dict[str, Any], key: str) -> None:
    with pytest.raises(Refusal) as caught:
        recipes.resolve(cfg, _unit(), overrides)
    assert caught.value.findings[0].key == key


def test_conflict_refused(cfg: Config) -> None:
    with pytest.raises(Refusal) as caught:
        recipes.resolve(cfg, _unit(), {"add": ["-O1"], "omit": ["-O1"]})
    assert caught.value.findings[0].key == "recipe.conflict"


def test_unit_omit_missing_refused(cfg: Config) -> None:
    with pytest.raises(Refusal) as caught:
        recipes.resolve(cfg, _unit(omit=["-O3"]), {})
    assert caught.value.findings[0].key == "recipe.option"


def test_overrides_that_repeat_the_units_own_options_resolve_as_the_unit_alone(cfg: Config) -> None:
    unit = _unit(add=["-mgp32"], omit=["-O2"])  # move and publish pass unit.options as the overrides
    assert recipes.resolve(cfg, unit, unit.options) == recipes.resolve(cfg, unit, {})


def test_a_units_omit_that_is_missing_names_the_unit(cfg: Config) -> None:
    with pytest.raises(Refusal) as caught:
        recipes.resolve(cfg, _unit("src/x.c", omit=["-O3"]), {})
    assert caught.value.findings[0].unit == "src/x.c"


def test_digest_changes_only_with_flags(cfg: Config) -> None:
    one = recipes.resolve(cfg, _unit("src/a.c"), {})
    two = recipes.resolve(cfg, replace(_unit("src/b.c"), members=("x", "y")), {})
    assert one.digest == two.digest
    changed = recipes.resolve(cfg, _unit(), {"omit": ["-O2"], "add": ["-O1"]})
    assert changed.digest != one.digest


def test_a_unit_preprocessor_option_goes_to_the_preprocessor_and_codegen_stays_with_the_compiler(cfg: Config) -> None:
    cfg = _config(cfg, cppflags=["-DGBI", "-undef"])
    for toolchain in ("gcc-test", "ido-7.1"):
        recipe = recipes.resolve(cfg, _unit(toolchain=toolchain, add=["-UGBI", "-G8"]), {})
        assert recipe.cppflags[-3:] == ("-DGBI", "-undef", "-UGBI") and "-UGBI" not in recipe.cflags
        assert "-G8" in recipe.cflags
    assert recipes.resolve(cfg, _unit(), {"omit": ["-DGBI"]}).cppflags[-1] == "-undef"


def test_gcc_rows_preprocess_as_they_always_did_and_only_ido_moved_to_the_shared_preprocessor() -> None:
    rows = configuration.load_resource("toolchains.toml")["toolchain"]
    shared = ["{cpp}", "{cppflags}", "{defines}", "{includes}", "{source}"]
    assert {i: r["preprocess"] for i, r in rows.items() if r["family"] == "gcc"} == {
        "gcc-2.7.2-kmc": shared,
        "gcc-2.8.1-sn64": shared,
    }
    assert {r["preprocess"][0] for i, r in rows.items() if r["family"] == "ido"} == {"{cpp}"}


def test_the_registrys_small_data_sizes_are_admissible_and_replace_the_old_one(
    cfg: Config, toolchains: dict[str, Any], paired: str
) -> None:
    toolchains["toolchain"][paired]["small_data"] = [0, 8]
    unit = _unit(toolchain=paired)
    for text, flags in (("-G8", ("-O2", "-G8")), ("-G 8", ("-O2", "-G", "8"))):
        assert recipes.resolve(cfg, unit, {"add": [text]}).cflags == flags
    with pytest.raises(Refusal) as error:
        recipes.resolve(cfg, unit, {"add": ["-G5"]})
    assert error.value.findings[0].key == "recipe.option"
