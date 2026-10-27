"""One header per source file: the pure hoist over plain texts."""

from __future__ import annotations

from unbake import hoist

TYPES = "#ifndef TYPES_H\n#define TYPES_H\ntypedef int s32;\ntypedef struct Foo { s32 a; } Foo;\n#endif\n"
CAMERA_H = '#ifndef CAMERA_H\n#define CAMERA_H\n#include "types.h"\nextern s32 Camera_old(s32 a);\n#endif\n'
CAMERA_C = (
    '#include "types.h"\n#include "CAMERA/CAMERA.H"\nextern s32 Camera_use(Foo *f, s32 n);\n'
    "s32 Camera_use(Foo *f, s32 n) { return f->a + n; }\n"
    "s32 Camera_old(s32 a) { return a; }\n"
    "s32 gCameraMode = 3;\n"
    "static s32 hidden(void) { return 1; }\n"
)
USER_C = (
    '#include "types.h"\nextern s32 Camera_use(Foo *f, s32 n);\nextern s32 gCameraMode;\n'
    "s32 other(Foo *f) { return Camera_use(f, gCameraMode); }\n"
)
TRACKED = ["include/types.h", "include/CAMERA/CAMERA.H", "src/game/CAMERA/CAMERA.CPP", "src/game/CAMERA/USER.CPP"]


def texts(**over: str) -> dict[str, str]:
    base = {
        "include/types.h": TYPES,
        "include/CAMERA/CAMERA.H": CAMERA_H,
        "src/game/CAMERA/CAMERA.CPP": CAMERA_C,
        "src/game/CAMERA/USER.CPP": USER_C,
    }
    return {**base, **over}


def test_exports_go_to_the_header_the_unit_already_includes() -> None:
    done = hoist.hoist(texts(), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert done.header == "include/CAMERA/CAMERA.H" and not done.created
    assert done.exported == ["Camera_use", "gCameraMode"]
    assert "extern s32 Camera_use(Foo *f, s32 n);" in done.files[done.header]
    assert "extern s32 gCameraMode;" in done.files[done.header]
    assert "hidden" not in done.files[done.header]


def test_users_lose_their_declarations_and_gain_the_include() -> None:
    done = hoist.hoist(texts(), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    user = done.files["src/game/CAMERA/USER.CPP"]
    assert "extern s32 Camera_use" not in user and "extern s32 gCameraMode" not in user
    assert '#include "CAMERA/CAMERA.H"' in user
    assert done.removed["src/game/CAMERA/USER.CPP"] == 2


def test_unit_externs_never_rise() -> None:
    done = hoist.hoist(texts(), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert done.externs_after <= done.externs_before


def test_a_header_is_created_when_the_unit_includes_none_of_its_name() -> None:
    other = CAMERA_C.replace('#include "CAMERA/CAMERA.H"\n', "")
    tree = texts(**{"src/game/CAMERA/CAMERA.CPP": other})
    del tree["include/CAMERA/CAMERA.H"]
    tracked = [t for t in TRACKED if t != "include/CAMERA/CAMERA.H"] + ["include/CAMERA/OTHER.H"]
    tree["include/CAMERA/OTHER.H"] = TYPES
    done = hoist.hoist(tree, tracked, "src/game/CAMERA/CAMERA.CPP")
    assert done.created and done.header == "include/CAMERA/CAMERA.H"
    assert done.files[done.header].startswith("#ifndef ")


def test_a_different_declaration_of_a_header_function_is_kept_and_listed() -> None:
    odd = USER_C + "extern s32 Camera_old(Foo *f);\n"
    done = hoist.hoist(texts(**{"src/game/CAMERA/USER.CPP": odd}), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert any(h.get("reason") == "differs" and h["path"] == "src/game/CAMERA/USER.CPP" for h in done.held)
    assert "extern s32 Camera_old(Foo *f);" in done.files.get("src/game/CAMERA/USER.CPP", odd)


def test_an_export_whose_type_the_header_cannot_see_is_held() -> None:
    c = (
        CAMERA_C.replace('#include "types.h"\n', "typedef struct Bar { int x; } Bar;\n")
        + "s32 take(Bar *b) { return b->x; }\n"
    )
    done = hoist.hoist(texts(**{"src/game/CAMERA/CAMERA.CPP": c}), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert any(h["name"] == "take" and h["reason"] == "type-unseen" for h in done.held)


def test_include_asm_functions_are_exported_when_every_declaration_agrees() -> None:
    c = CAMERA_C + 'INCLUDE_ASM("asm/nonmatchings/CAMERA", Camera_asm);\n'
    user = USER_C + "extern void Camera_asm(s32 a);\n"
    done = hoist.hoist(
        texts(**{"src/game/CAMERA/CAMERA.CPP": c, "src/game/CAMERA/USER.CPP": user}),
        TRACKED,
        "src/game/CAMERA/CAMERA.CPP",
    )
    assert "Camera_asm" in done.exported


def test_a_statement_cut_inside_a_macro_call_is_left_alone() -> None:
    c = CAMERA_C + "MACRO_CALL(a, b\n;\n"
    done = hoist.hoist(texts(**{"src/game/CAMERA/CAMERA.CPP": c}), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert "Camera_use" in done.exported


def test_an_export_a_user_declares_differently_stays_out_of_the_header() -> None:
    odd = USER_C.replace("Foo *f, s32 n", "Foo *f")
    done = hoist.hoist(texts(**{"src/game/CAMERA/USER.CPP": odd}), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert "Camera_use" not in done.exported
    assert any(h["name"] == "Camera_use" and h["reason"] == "users-differ" for h in done.held)


def test_a_forward_declaration_is_no_export_and_no_header_is_made_for_nothing() -> None:
    c = "struct Tag;\nextern s32 Camera_use(struct Tag *t);\nstatic s32 hidden(void) { return 1; }\n"
    tree = texts(**{"src/game/CAMERA/CAMERA.CPP": c})
    del tree["include/CAMERA/CAMERA.H"]
    done = hoist.hoist(tree, [t for t in TRACKED if t != "include/CAMERA/CAMERA.H"], "src/game/CAMERA/CAMERA.CPP")
    assert done.exported == [] and done.files == {}


def test_the_declaration_goes_with_its_trailing_comment() -> None:
    user = USER_C.replace("extern s32 gCameraMode;", "extern s32 gCameraMode;  /* extern */")
    done = hoist.hoist(texts(**{"src/game/CAMERA/USER.CPP": user}), TRACKED, "src/game/CAMERA/CAMERA.CPP")
    assert "/* extern */" not in done.files["src/game/CAMERA/USER.CPP"]


def test_the_macro_header_is_never_the_include_of_a_hoisted_header() -> None:
    macro = '#ifndef M_H\n#define M_H\n#include "types.h"\n#define INCLUDE_ASM(S, N) __asm__(#N)\n#endif\n'
    c = CAMERA_C.replace('#include "types.h"\n', '#include "include_asm.h"\n')
    tree = texts(**{"src/game/CAMERA/CAMERA.CPP": c, "include/include_asm.h": macro})
    done = hoist.hoist(tree, [*TRACKED, "include/include_asm.h"], "src/game/CAMERA/CAMERA.CPP")
    assert "include_asm.h" not in done.files.get(done.header, "")


def test_the_private_header_of_a_merged_unit_is_folded_into_the_header() -> None:
    old_h = '#ifndef OLD_H\n#define OLD_H\n#include "types.h"\nextern s32 Camera_use(Foo *f, s32 n);\n#endif\n'
    c = CAMERA_C.replace('#include "CAMERA/CAMERA.H"\n', '#include "CAMERA/OLD.H"\n')
    tree = texts(**{"src/game/CAMERA/CAMERA.CPP": c, "include/CAMERA/OLD.H": old_h})
    done = hoist.hoist(tree, [*TRACKED, "include/CAMERA/OLD.H"], "src/game/CAMERA/CAMERA.CPP")
    assert done.deleted == ["include/CAMERA/OLD.H"]
    assert '#include "CAMERA/CAMERA.H"' in done.files["src/game/CAMERA/CAMERA.CPP"]
    assert "OLD.H" not in done.files["src/game/CAMERA/CAMERA.CPP"]
    assert "extern s32 Camera_use(Foo *f, s32 n);" in done.files[done.header]
    assert done.externs_after <= done.externs_before


def test_a_header_other_units_include_is_not_folded() -> None:
    old_h = '#ifndef OLD_H\n#define OLD_H\n#include "types.h"\nextern s32 Camera_use(Foo *f, s32 n);\n#endif\n'
    c = CAMERA_C.replace('#include "CAMERA/CAMERA.H"\n', '#include "CAMERA/OLD.H"\n')
    u = USER_C.replace('#include "types.h"\n', '#include "types.h"\n#include "CAMERA/OLD.H"\n')
    tree = texts(**{"src/game/CAMERA/CAMERA.CPP": c, "src/game/CAMERA/USER.CPP": u, "include/CAMERA/OLD.H": old_h})
    done = hoist.hoist(tree, [*TRACKED, "include/CAMERA/OLD.H"], "src/game/CAMERA/CAMERA.CPP")
    assert done.deleted == []
