"""Type evidence tests with all native and worker boundaries mocked."""

import tomllib
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import fixture
import pytest
import tomlkit

from unbake import config, types
from unbake.contracts import Finding, LayoutMap, Member, Placement, Refusal, Snapshot

AGREED: dict = {}  # what headers.disagreements answers: name -> the one type its declarations agree on
CONFLICTS: list = []
LANDED: dict = {}  # what headers.landed answers: name -> the head of a landed definition
HEADER_DECLARATIONS: dict = {}  # what headers.catalog answers: name -> (path, line, statement)


def _document(**tables):
    if "global_" in tables:
        tables["global"] = tables.pop("global_")
    return {"schema": 1, "function": {}, "global": {}, "struct": {}, **tables}


def _with_doc(snapshot, doc):
    return replace(snapshot, overlays={**snapshot.overlays, "types.toml": tomlkit.dumps(doc).encode()})


def _member(name, kind="function", holders=("a", "b"), section=".text", size=8):
    placements = tuple(
        Placement(
            v,
            section,
            0x1000,
            0x1000 if section == ".bss" else 0x1000 + size,
            0x80000400,
            size if section == ".bss" else 0,
        )
        for v in holders
    )
    return Member(name, kind, "asm" if kind == "function" else "data", "group", placements)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    cfg = fixture.bare_config(tmp_path, name="test", title="Test", toolchain="test-tc", author=("test", "test@invalid"))
    members = {name: _member(name) for name in ("alpha", "zeta")}
    members["buffer"] = _member("buffer", "data", section=".data", size=0x18)
    snapshot = Snapshot(cfg, "head", LayoutMap({}, members, {}, "layout", (), {}), {}, {}, "snapshot")
    overlays = {
        f"asm/{v}/{name}.s": f"glabel {name}\n nop\n".encode() for name in ("alpha", "zeta") for v in ("a", "b")
    }
    snapshot = replace(
        _with_doc(snapshot, _document()),
        overlays={**_with_doc(snapshot, _document()).overlays, **overlays, "include/h.h": b"header"},
    )
    span = Mock()

    @contextmanager
    def stage(_name):
        yield span

    cache = {}

    def cached(config, kind, key, produce):
        assert config is cfg
        if (kind, key) not in cache:
            cache[kind, key] = produce()
        return cache[kind, key]

    resources = {"units.toml": {"kind": {"c": {"decompiled": True}, "asm": {"decompiled": False}}}}
    monkeypatch.setattr(types.effort, "stage", stage)
    monkeypatch.setattr(types.store, "cached", cached)
    monkeypatch.setattr(
        types.headers, "sources", lambda snap, place="include", suffix=".h": ["include/h.h"] * (place == "include")
    )
    monkeypatch.setattr(types.headers, "disagreements", lambda snap, defined=(): (AGREED, CONFLICTS))
    monkeypatch.setattr(types.headers, "catalog", lambda snap: HEADER_DECLARATIONS)
    monkeypatch.setattr(types.headers, "landed", lambda snap: LANDED)
    HEADER_DECLARATIONS.clear()
    LANDED.clear()
    AGREED.clear()
    CONFLICTS.clear()
    monkeypatch.setattr(types.config, "load_resource", Mock(side_effect=resources.__getitem__))
    return (snapshot,)


def test_scan_takes_the_type_the_landed_declarations_agree_on_and_only_in_types_it_knows(lane):
    snapshot, *_ = lane
    HEADER_DECLARATIONS["Word"] = ("include/h.h", 1, "typedef int Word;")
    AGREED.update({"alpha": "void @(Word)", "zeta": "Local @(int)"})
    members = {"alpha": _member("alpha"), "zeta": _member("zeta")}
    snapshot = replace(snapshot, layout=replace(snapshot.layout, members=members))
    doc = tomllib.loads(types.scan(snapshot).decode())
    assert doc["function"]["alpha"] == {"signature": "void alpha(Word)", "evidence": "declared"}
    assert "zeta" not in doc["function"]  # Local is a type no header gives the type map


def test_conflicts_are_the_findings_of_the_names_the_declarations_disagree_on(lane):
    snapshot, *_ = lane
    CONFLICTS.append("finding")
    assert types.conflicts(_with_doc(snapshot, _document())) == ["finding"]


def test_scan_records_the_headers_declarations_as_authored(lane):
    snapshot, *_ = lane
    HEADER_DECLARATIONS.update(
        {
            "struct Pair": ("include/h.h", 3, "struct Pair { int a; int b; };"),
            "Word": ("include/h.h", 4, "typedef int Word;"),
            "gCount": ("include/h.h", 5, "extern int gCount;"),
            "handler": ("include/h.h", 6, "extern void (*handler)(int a);"),
            "known": ("include/h.h", 7, "extern int known(int a, void *b);"),
            "bare": ("include/h.h", 8, "void bare(void);"),
        }
    )
    LANDED["known"] = "s32 known(s32 a, void *b)"
    doc = tomllib.loads(types.scan(snapshot).decode())
    assert doc["struct"] == {
        "Pair": {"declaration": "struct Pair { int a; int b; };", "evidence": "authored"},
        "Word": {"declaration": "typedef int Word;", "evidence": "authored"},
    }
    assert doc["global"] == {
        "gCount": {"declaration": "extern int gCount;", "evidence": "authored"},
        "handler": {"declaration": "extern void (*handler)(int a);", "evidence": "authored"},
    }
    assert doc["function"] == {
        "known": {"signature": "s32 known(s32 a, void *b)", "evidence": "landed"},  # outranks
        "bare": {"signature": "void bare(void)", "evidence": "authored"},
    }
    assert "splat" not in str(doc)
    config.validate("types", doc, "types.toml")


def test_scan_deterministic(lane):
    snapshot, *_ = lane
    first = types.scan(snapshot)
    reordered = replace(
        snapshot, layout=replace(snapshot.layout, members=dict(reversed(list(snapshot.layout.members.items()))))
    )
    assert types.scan(reordered) == first


def test_authority_lays_the_landed_signatures_over_the_stored_map(lane):
    snapshot, *_ = lane
    LANDED["known"] = "s32 known(s32 a)"
    stored = types.load(snapshot)
    got = types.authority(snapshot)
    assert got["function"]["known"] == {"signature": "s32 known(s32 a)", "evidence": "landed"}
    assert "known" not in stored["function"] or stored["function"]["known"] != got["function"]["known"]
    assert got["global"] == stored["global"] and got["struct"] == stored["struct"]


def test_load_missing_refuses(lane):
    snapshot, *_ = lane
    snapshot = replace(snapshot, overlays={"types.toml": None})
    with pytest.raises(Refusal) as caught:
        types.load(snapshot)
    assert caught.value.findings == (
        Finding("config.missing", "types.toml does not exist", path="types.toml", action="unbake setup"),
    )


def test_load_validates(lane, monkeypatch):
    snapshot, *_ = lane
    validate = Mock()
    monkeypatch.setattr(types.config, "validate", validate)
    assert types.load(snapshot) == _document()
    validate.assert_called_once_with("types", _document(), "types.toml")


def test_types_toml_is_parsed_once_per_command(lane, monkeypatch):
    from unbake import effort

    snapshot = lane[0] if isinstance(lane, tuple) else lane
    parses = []
    real = types.config.tomllib.loads
    monkeypatch.setattr(types.config.tomllib, "loads", lambda text: parses.append(1) or real(text))
    with effort.command("check", []):
        assert types.load(snapshot) is types.load(snapshot)
    assert len(parses) == 1


def _use(*stores, calls=()):
    accesses = [(a, 4, "sw", "based", a) for a in stores] + [(a, 0, "jal", "taken", None) for _, ((_, a), *_) in calls]
    return {"accesses": accesses, "calls": list(calls), "args": [], "steps": []}


def test_a_fill_and_two_stores_are_one_span_with_the_writer_and_far_stores_are_two():
    near = _use(0x100, 0x108, calls=[("fill", ((0, 0x200), (2, 0x40)))])
    assert types._spans(near) == [[0x100, 0x240, 2, 1]]
    far = _use(0x100, 0x104, 0x108, 0x10C, 0x2000, 0x2004, 0x2008, 0x200C)
    assert types._spans(far) == [[0x100, 0x110, 4, 0], [0x2000, 0x2010, 4, 0]]
    assert types._spans(_use(0x100, 0x104, 0x108)) == []  # three stores and no fill are not a run


def test_usage_rows_name_the_readers_spelling_span_base_and_stride(lane, monkeypatch):
    snapshot, *_ = lane
    members = {"w": replace(_member("w"), state="c"), "r": _member("r")}
    versions = {"a": Mock(symbols={"obj": 0x100, "next": 0x200}, rom_sha256="x")}
    snapshot = replace(snapshot, versions=versions, layout=replace(snapshot.layout, members=members))
    uses = {
        "w": _use(0x100, 0x104, 0x108, 0x10C) | {"steps": [(0x100, 0xC)]},
        "r": {"accesses": [(0x104, 4, "lw", "based", 0x100)], "calls": [], "args": [(0, 8, 4)], "steps": []},
    }
    readers = {0x104: [("w", "based", 4, 0x100), ("r", "based", 4, 0x100)], 0x100: [("w", "based", 4, 0x100)]}
    monkeypatch.setattr(types, "_tables", lambda s, v: (uses, readers))
    monkeypatch.setattr(types.headers, "externs", lambda s: {"obj": [("src/w.c", 3, "Thing *")]})
    rows = types.usage_at(snapshot, "obj")["a"]
    assert [r["address"] for r in rows] == [0x100, 0x104]
    row = rows[1]
    assert row["readers"] == [("w", True), ("r", False)] and row["landed_readers"] == 1
    assert row["span"] == {"start": 0x100, "size": 0x10, "writer": "w", "stores": 4, "fills": 0}
    assert row["base"] == ("obj", 0) and row["stride"] == 0xC and rows[0]["spellings"] == [("src/w.c", 3, "Thing *")]
    assert types.usage(snapshot, "r", "a")["args"] == {"a0": {"offsets": [8], "stride": None, "landed_callers": []}}


def _record(source, name="struct R"):
    from unbake import headers
    from unbake.families import layoutattr

    text, tainted = layoutattr.rewrite(source)
    return types.record_layouts(headers.parse(text, "h"), [name], tainted)[name]


def test_packed_record_layout():
    row = _record("struct R { short a; int b; } __attribute__((packed));")
    assert [(f["name"], f["offset"]) for f in row["fields"]] == [("a", 0), ("b", 2)]
    assert (row["size"], row["alignment"]) == (6, 1)


def test_pragma_pack_2_caps_alignment():
    row = _record("#pragma pack(push, 2)\nstruct R { char a; int b; double c; };\n#pragma pack(pop)\n")
    assert [f["offset"] for f in row["fields"]] == [0, 2, 6]
    assert (row["size"], row["alignment"]) == (14, 2)


def test_aligned_field_inside_packed():
    row = _record("struct R { char a; int b __attribute__((aligned(4))); } __attribute__((packed));")
    assert [f["offset"] for f in row["fields"]] == [0, 4]


def test_scan_keeps_authored_rows_the_c_does_not_produce_and_drops_stale_landed_and_declared_ones(lane):
    snapshot, *_ = lane
    doc = _document(
        function={
            "func_asm_only": {"signature": "void func_asm_only(void)", "evidence": "authored"},
            "gone": {"signature": "void gone(void)", "evidence": "landed"},
            "was_declared": {"signature": "void was_declared(void)", "evidence": "declared"},
        },
        struct={"Hand": {"declaration": "struct Hand { int a; };", "evidence": "authored"}},
        global_={"gHand": {"declaration": "extern int gHand;", "evidence": "authored"}},
    )
    out = tomllib.loads(types.scan(_with_doc(snapshot, doc)).decode())
    assert out["function"] == {"func_asm_only": {"signature": "void func_asm_only(void)", "evidence": "authored"}}
    assert out["struct"]["Hand"]["evidence"] == "authored" and out["global"]["gHand"]["evidence"] == "authored"
    config.validate("types", out, "types.toml")


def test_scanning_the_scanned_map_changes_nothing_so_setup_generate_and_land_all_keep_the_rows(lane):
    snapshot, *_ = lane
    doc = _document(function={"asm_only": {"signature": "void asm_only(void)", "evidence": "authored"}})
    first = types.scan(_with_doc(snapshot, doc))
    again = replace(snapshot, overlays={**snapshot.overlays, "types.toml": first})
    assert types.scan(again) == first and b"asm_only" in first
