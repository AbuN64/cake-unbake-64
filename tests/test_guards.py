"""Generated guard bindings: checked once before layout writes, no compiler or ROM."""

from types import SimpleNamespace

import pytest

from unbake import api, folders, guards
from unbake.contracts import Refusal

HEADER = "#ifndef CAMERA_80233920_31_H\n#define CAMERA_80233920_31_H\nvoid Gsky_GetFloorSurfaceShake(void);\n#endif\n"
PATH = "include/CAMERA/GSKY_GETFLOORSURFACESHAKE.H"


def test_content_identity_keeps_every_byte_except_two_wrapper_tokens():
    old = '#ifndef OLD_H\n#define OLD_H\n#include "owner.h"\nextern float D_800C9FEC;\n#endif\n'
    new = old.replace("OLD_H", "OWNER_HEALTH_H")
    assert api.guard_content(old) == api.guard_content(new)
    assert api.guard_content(old) is not None
    for changed in (
        new.replace("float", "int"),
        new.replace("owner.h", "different.h"),
        new.replace("extern", "/* retained */ extern"),
        new.replace("extern", "#if ACTIVE\nextern").replace("#endif", "#endif\n#endif"),
    ):
        assert api.guard_content(old) != api.guard_content(changed)
    for invalid in (
        old.replace("extern", "int OLD_H;\nextern"),
        old.replace("#endif", "#else\n#endif"),
        old + "int outside;\n",
    ):
        assert api.guard_content(invalid) is None


def test_real_renamed_header_and_all_defined_consumers_preserve_activation_and_literals():
    old = "CAMERA_80233920_31_H"
    consumer = f'#if defined({old}) && defined(OTHER)\n#include "x.h"\n#endif\n'
    texts = {PATH: HEADER, "src/a.c": consumer + f'/* {old} */\nconst char *s = "{old}";\n'}
    got = guards.rewrite(texts, {PATH: PATH})
    new = api.guard_name(PATH)
    assert guards.binding(got[PATH]) == new
    assert f"defined({new})" in got["src/a.c"]
    assert f"/* {old} */" in got["src/a.c"] and f'"{old}"' in got["src/a.c"]
    assert guards.rewrite({**texts, **got}, {PATH: PATH}) == {}
    assert texts[PATH] == HEADER


@pytest.mark.parametrize(
    "extra",
    [
        {"include/other.h": HEADER},
        {"src/a.c": "#define CAMERA_80233920_31_H 1\n"},
        {"src/a.c": "#undef CAMERA_80233920_31_H\n"},
        {"src/a.c": "#if CAMERA_80233920_31_H\n#endif\n"},
        {"src/a.c": "int CAMERA_80233920_31_H;\n"},
        {"src/a.c": "#define COPY CAMERA_80233920_31_H\n"},
        {"config.toml": 'cppflags = ["-DCAMERA_80233920_31_H"]\n'},
        {"src/a.c": "int CAMERA_GSKY_GETFLOORSURFACESHAKE_H;\n"},
    ],
)
def test_shared_colliding_or_ambiguous_consumers_refuse(extra):
    with pytest.raises(ValueError):
        guards.rewrite({PATH: HEADER, **extra}, {PATH: PATH})


@pytest.mark.parametrize(
    "text",
    [
        HEADER + "int outside;\n",
        HEADER.replace("#endif", "#else\nint alternate;\n#endif"),
        HEADER.replace("#define CAMERA_80233920_31_H", "#define CAMERA_80233920_31_H 1"),
        "#pragma once\nint a;\n",
    ],
)
def test_partial_or_value_guards_are_not_silently_normalized(text):
    assert guards.binding(text) is None
    with pytest.raises(ValueError):
        guards.rewrite({PATH: text}, {PATH: PATH})


def test_public_header_action_batches_identity_map_and_refuses_before_any_write(tmp_path, monkeypatch):
    (tmp_path / PATH).parent.mkdir(parents=True)
    (tmp_path / PATH).write_text(HEADER)
    other = "include/other.h"
    (tmp_path / other).write_text(HEADER)
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(folders, "_tracked", lambda _: [PATH, other])
    selected = tmp_path / "reviewed.tsv"
    selected.write_text(f"{PATH}\t{PATH}\n")
    with pytest.raises(Refusal):
        folders.header(config, {"map": str(selected)})
    assert (tmp_path / PATH).read_text() == HEADER
    (tmp_path / other).write_text("#pragma once\n")
    got = folders.header(config, {"map": str(selected)})
    assert got["written"] == [PATH]
    assert folders.header(config, {"map": str(selected)})["written"] == []


def test_batch_collects_shared_owners_consumers_and_destinations_without_writes(tmp_path, monkeypatch):
    shared = "#ifndef SYSTEM_2_802661FC_H\n#define SYSTEM_2_802661FC_H\n#endif\n"
    texts = {
        "include/SYSTEM/802661FC.H": shared,
        "include/SYSTEM/CODE_802661FC.H": shared,
        PATH: HEADER,
        "src/consumer.c": "#undef CAMERA_80233920_31_H\nint CAMERA_GSKY_GETFLOORSURFACESHAKE_H;\n",
        "config.toml": 'cppflags = ["-DSYSTEM_2_802661FC_H", "-UCAMERA_80233920_31_H"]\n',
    }
    for path, body in texts.items():
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(folders, "_tracked", lambda _: list(texts))
    selection = tmp_path / "guards.tsv"
    selection.write_text(f"{PATH}\t{PATH}\ninclude/SYSTEM/CODE_802661FC.H\tinclude/SYSTEM/CODE_802661FC.H\n")
    with pytest.raises(Refusal) as caught:
        folders.header(config, {"map": str(selection)})
    reasons = [f.reason for f in caught.value.findings]
    assert len(reasons) == 5
    assert any("shared guard SYSTEM_2_802661FC_H" in r for r in reasons)
    assert any("already used" in r for r in reasons)
    assert any("in #undef" in r for r in reasons)
    assert sum("outside C/preprocessor" in r for r in reasons) == 2
    assert all((tmp_path / path).read_text() == body for path, body in texts.items())
