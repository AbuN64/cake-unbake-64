"""The packaged SDK signature catalog: generic tool data, valid, and free of anything about one game."""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

from unbake import config as configuration

DENY = ("ragewars", "battletanx", "turok")  # games the tool's own data must never name
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def catalog() -> dict[str, Any]:
    return json.loads((resources.files("unbake.resources") / "data" / "sdk" / "catalog.json").read_bytes())  # type: ignore[no-any-return]


def test_catalog_validates_against_its_schema() -> None:
    document = catalog()
    configuration.validate("sdk", document, "sdk/catalog.json")
    assert document["signatures"] and {r["library"] for r in document["signatures"]} >= {"libgultra", "libultra"}


def test_catalog_names_no_game_path_or_address() -> None:
    text = json.dumps(catalog()).lower()
    assert not [word for word in DENY if word in text]
    clean = not any(
        mark in text for mark in ("/", "\\", "0x8")
    )  # never inside an assert: a failure would print the text
    assert clean
    for row in catalog()["signatures"]:
        assert (
            NAME.match(row["name"])
            and NAME.match(row["module"])
            and row["library"] in {"libultra", "libgultra", "libgcc", "libm"}
        )
        assert set(row) - {"name", "module", "library", "versions"} in (
            {"size", "crc_head", "crc_body"},
            {"words", "masks"},
        )


def test_catalog_has_no_code_bytes_beyond_short_masked_words() -> None:
    for row in catalog()["signatures"]:
        if "words" in row:
            assert len(row["words"]) < 8 and len(row["words"]) == len(row["masks"])
