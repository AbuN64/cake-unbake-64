"""delta: a map over every unit derives again only the items whose recorded inputs moved."""

from types import SimpleNamespace

import pytest

from unbake import delta

ITEMS = ("one", "two", "three")
READS = {"one": ("include/a.h", "src/one.c"), "two": ("include/b.h", "src/two.c"), "three": ("include/a.h",)}


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """Files as pins, a stamp per item and a pool that runs every job it is given (no cache: a dispatch is a job)."""
    state = SimpleNamespace(
        pins={f: "1" for files in READS.values() for f in files} | {"include/missing.h": "missing"},
        stamps=dict.fromkeys(ITEMS, "s"),
        jobs=[],
        during={},  # a file rewritten while the job read it: the pin the job saw
        code="tool-1",
        reads=dict(READS),
    )
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))

    def mapped(cfg, name, function, items, key=None):
        state.jobs.extend(items)
        return [function(item) for item in items]

    def job(item):
        return f"rule:{item}", {f: state.during.get(f, state.pins[f]) for f in state.reads[item]}

    monkeypatch.setattr(delta.pool, "map", mapped)
    monkeypatch.setattr(delta, "code_identity", lambda function: state.code)

    def run(items=ITEMS, code=True):
        return delta.map(
            config,
            "rules",
            job,
            list(items),
            None,
            ident=lambda item: item,
            stamp=lambda item: state.stamps[item],
            pin=lambda f: state.pins.get(f, "missing"),
            code=code,
        )

    state.run = run
    return state


def test_a_second_run_over_the_same_inputs_dispatches_nothing(scene):
    assert scene.run() == ["rule:one", "rule:two", "rule:three"]
    assert scene.run() == ["rule:one", "rule:two", "rule:three"] and scene.jobs == list(ITEMS)


def test_a_header_edit_derives_only_the_items_that_read_it(scene):
    """A one-item landing rendered the rules of all 22,905 units: only the units reaching the edited file go again."""
    scene.run()
    scene.jobs.clear()
    scene.pins["include/b.h"] = "2"
    scene.run()
    assert scene.jobs == ["two"]
    scene.jobs.clear()
    scene.pins["include/a.h"] = "2"
    scene.run()
    assert scene.jobs == ["one", "three"]


def test_a_file_created_where_a_walk_found_none_derives_the_items_that_probed_it(scene):
    scene.reads["two"] = (*READS["two"], "include/missing.h")
    scene.run()
    scene.jobs.clear()
    scene.pins["include/missing.h"] = "now"
    scene.run()
    assert scene.jobs == ["two"]


def test_a_stamp_change_derives_only_that_item(scene):
    scene.run()
    scene.jobs.clear()
    scene.stamps["three"] = "moved row"
    scene.run()
    assert scene.jobs == ["three"]


def test_a_new_item_is_derived_and_a_gone_one_costs_nothing(scene):
    scene.run(ITEMS[:2])
    scene.jobs.clear()
    assert scene.run() == ["rule:one", "rule:two", "rule:three"] and scene.jobs == ["three"]
    scene.jobs.clear()
    scene.run(ITEMS[1:])
    assert scene.jobs == []


def test_a_record_of_inputs_alone_survives_a_tool_install(scene):
    """A tool install keyed all 269 fuzzy rows again (5 min in one landing): such a record ignores the code."""
    scene.run(code=False)
    scene.jobs.clear()
    scene.code = "tool-2"
    scene.run(code=False)
    assert scene.jobs == []


def test_a_record_of_other_tool_code_is_not_used_by_a_code_bound_map(scene):
    scene.run()
    scene.jobs.clear()
    scene.code = "tool-2"
    scene.run()
    assert scene.jobs == list(ITEMS)


def test_a_file_rewritten_while_its_job_read_it_is_derived_again(scene):
    scene.during["src/two.c"] = "half-written"
    scene.run()
    scene.during.clear()
    scene.jobs.clear()
    scene.run()
    assert scene.jobs == ["two"]


def test_a_code_bound_record_is_keyed_by_the_jobs_code_not_the_whole_install(scene, monkeypatch):
    """An install that leaves the job's code alone stays warm."""
    asked = []
    monkeypatch.setattr(delta, "code_identity", lambda function: asked.append(function) or "same job code")
    monkeypatch.setattr(delta.store, "code", lambda: 1 / 0)  # the install digest is never read
    scene.run()
    scene.jobs.clear()
    scene.run()
    assert scene.jobs == [] and asked and all(callable(f) for f in asked)
