from copy import deepcopy

import pytest

from unbake.contracts import Refusal
from unbake.symbols import names_plan as plan


def document():
    return {
        "schema": 1,
        "functions": [{"name": "Original", "version": "v", "address": 0x80001000}],
        "structs": [],
        "typedefs": [],
        "enums": [],
    }


def test_exact_address_and_no_guess_for_unmatched():
    data = document()
    tables = {"v": {"func_old": (0x80001000, None, None, "func")}}
    assert plan(data, tables, {})["moves"] == [{"op": "symbol", "subject": "func_old", "target": "Original"}]
    data["functions"][0]["address"] += 4
    assert plan(data, tables, {})["moves"] == []
    assert plan(data, tables, {})["unmatched"][0]["kind"] == "function"


def test_conflicting_names_refuse_whole_plan():
    data = document()
    data["functions"].append({**data["functions"][0], "name": "Other"})
    with pytest.raises(Refusal):
        plan(data, {"v": {"old": (0x80001000, None, None, "func")}}, {})


def test_data_address_is_not_a_function_and_collision_refuses():
    data = document()
    assert not plan(data, {"v": {"data": (0x80001000, None, None, "data")}}, {})["moves"]
    with pytest.raises(Refusal):
        plan(data, {"v": {"old": (0x80001000, None, None, "func"), "Original": (0x80002000, None, None, "func")}}, {})


def test_type_binding_requires_every_offset_size_and_type():
    data = document()
    data["functions"] = []
    row = {
        "name": "Original_t",
        "version": "v",
        "kind": "struct_tag",
        "size": 4,
        "complete": True,
        "members": [{"name": "m_value", "offset": 0, "size": 4, "type": "int"}],
    }
    data["structs"] = [row]
    data["typedefs"] = [{"name": "Original", "version": "v", "type": {"tag": "Original_t", "modifiers": []}}]
    bound = {**deepcopy(row), "original": "Original_t", "name": "Current"}
    bound["members"][0]["name"] = "value"
    result = plan(data, {}, {"types": [bound]})
    assert result["moves"] == [
        {"op": "type", "subject": "Current", "target": "Original"},
        {"op": "type", "subject": "struct Original", "target": "struct Original_t"},
    ]
    assert result["unmatched"][0]["kind"] == "field"
    for key, value in [("offset", 1), ("size", 2), ("type", "float")]:
        bad = deepcopy(bound)
        bad["members"][0][key] = value
        assert not plan(data, {}, {"types": [bad]})["moves"]


def test_unsafe_version_refuses_before_path_use():
    data = document()
    data["functions"][0]["version"] = "../../outside"
    with pytest.raises(Refusal):
        plan(data, {}, {})


def test_command_renames_alias_facts_and_literal_owner_without_touching_files(tmp_path):
    import json

    from unbake.symbols import names_command as command

    version = tmp_path / "versions" / "v"
    version.mkdir(parents=True)
    (version / "symbol_addrs.txt").write_text("func_old = 0x80001000; // type:func\n")
    aliases = version / "aliases.txt"
    before = "literals_func_old = 0x80002000; // rom:0x2000\nfunc_old = 0x80001000; // type:func\n"
    aliases.write_text(before)
    source = tmp_path / "names.json"
    source.write_text(json.dumps(document()))
    result = command({"project": tmp_path, "input": source, "bindings": None})
    assert result["symbol_files"]["versions/v/aliases.txt"] == before.replace("func_old", "Original")
    assert aliases.read_text() == before
