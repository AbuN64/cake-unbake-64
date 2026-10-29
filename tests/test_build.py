"""The generated whole-ROM build files: Makefile, link script and splat yaml. Mocked: no ROM, compiler or make."""

import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unbake import adapters, build
from unbake.contracts import FileFacts, Refusal, Segment
from unbake.resources import cc_cache
from unbake.tree import Module, Tree

SHA = {"us": "a" * 40, "eu-1": "b" * 40}


def _pp(variable: str, command: str) -> str:
    """A make override that preprocesses with `command` as the real recipe does: the stamp's one process."""
    unit = f"{sys.executable} -I -S {Path(cc_cache.__file__)} --unit ''"
    return f"{variable}=@{unit} $@ $(call shq,$(PP_FLAGS)) $(HDRKEY) $(STAMPREC) {command}"


CCPY = f"CCPY={sys.executable} {Path(__file__).parent / 'ccpy_stub.py'}"  # the helper, its trim only logging


def segments(version: str) -> tuple[Segment, ...]:
    if version == "us":
        return (Segment("main", 0x1000, 0x80200400, "code"), Segment("assets", 0x2000, None, "bin"))
    return (Segment("main", 0x1000, 0x80200500, "code"),)


def make_config(tmp_path: Path, files: dict[str, FileFacts] | None = None) -> SimpleNamespace:
    (tmp_path / "roms").mkdir(exist_ok=True)
    for v in SHA:
        (tmp_path / "roms" / f"baserom.{v}.z64").write_bytes(bytes(0x3000))
        (tmp_path / f"versions/{v}").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"versions/{v}/symbol_addrs.txt").touch()
    project = SimpleNamespace(
        root=tmp_path,
        name="Game",
        title="Game",
        toolchain="gcc-2.8.1-sn64",
        cppflags=("-DF3DEX_GBI_2",),
        cflags=(),
        asflags=(),
        versions=("eu-1", "us"),
        version={v: SimpleNamespace(sha1=SHA[v], size=0x3000, segments=segments(v)) for v in SHA},
        files=files or {},
    )
    return SimpleNamespace(project=project, host=SimpleNamespace(workers=2))


def module(path: str, segment: str, sections: dict[str, str] | None = None, **owns: tuple[str, ...]) -> Module:
    """A module over names `f_N` (a function) or `d_N` (data in the section `sections` gives, default .data)."""
    held = {v.replace("_", "-"): names for v, names in owns.items()}
    kinds = {v: {n: (sections or {}).get(n, ".text" if n.startswith("f_") else ".data") for n in names}
             for v, names in held.items()}  # fmt: skip
    return Module(path, segment, held, (), {}, kinds)


def make_tree(*modules: Module, base: int = 0x1000) -> Tree:
    """Symbols give each name a rom address by its number, so a module's order is the order of its first name."""
    table: dict[str, dict[str, tuple[int, int, int, str]]] = {}
    for m in modules:
        for v, names in m.owns.items():
            for name in names:
                n = int(name.rsplit("_", 1)[1])
                kind = "func" if name.startswith("f_") else "data"
                table.setdefault(v, {})[name] = (0x80200000 + n, base + n * 0x10, 0x10, kind)
    return Tree("digest", modules, table)


@pytest.fixture
def per_object(monkeypatch):
    """The toolchain of make_config keeps a function's literals with its object (as gcc-2.7.2-kmc does)."""
    real = build.configuration.toolchain
    monkeypatch.setattr(build.configuration, "toolchain", lambda id: {**real(id), "link_sections": "per-object"})


@pytest.fixture
def no_apart(monkeypatch):
    """A toolchain row without function_sections: an object is one `.text` and one `.rodata`, whatever the ROM does."""
    real = build.configuration.toolchain
    monkeypatch.setattr(
        build.configuration, "toolchain", lambda id: {k: v for k, v in real(id).items() if k != "function_sections"}
    )


def both(path: str, *names: str, sections: dict[str, str] | None = None) -> Module:
    return module(path, "main", sections, us=names, eu_1=names)


@pytest.mark.usefixtures("per_object")
def test_linker_golden(tmp_path):
    config = make_config(tmp_path)
    tree = make_tree(both("src/a.c", "f_1"), module("src/b.c", "main", None, us=("f_2",)), both("src/c.c", "f_3"))
    assert build.linker(config, tree).decode() == "\n".join(
        [
            "/* generated from config.toml; do not edit */",
            "OUTPUT_ARCH(mips)",
            "SECTIONS",
            "{",
            "#if defined(VERSION_EU_1)",
            "    .main 0x80200500 : AT(0x1000) SUBALIGN(1)",
            "#elif defined(VERSION_US)",
            "    .main 0x80200400 : AT(0x1000) SUBALIGN(1)",
            "#endif",
            "    {",
            "        . = MAX(., 0x10);",
            "        src/a.c.o(.text);",
            "        src/a.c.o(.rodata .rdata);",
            "#if defined(VERSION_US)",
            "        . = MAX(., 0x20);",
            "        src/b.c.o(.text);",
            "        src/b.c.o(.rodata .rdata);",
            "#endif",
            "        . = MAX(., 0x30);",
            "        src/c.c.o(.text);",
            "        src/c.c.o(.rodata .rdata);",
            "    }",
            "    .main.bss (NOLOAD) :",
            "    {",
            "        src/a.c.o(.bss .sbss COMMON);",
            "#if defined(VERSION_US)",
            "        src/b.c.o(.bss .sbss COMMON);",
            "#endif",
            "        src/c.c.o(.bss .sbss COMMON);",
            "    }",
            "#if defined(VERSION_US)",
            "    .assets 0x00002000 : AT(0x2000) SUBALIGN(1)",
            "    {",
            "        assets/assets.bin.o(.data);",
            "    }",
            "#endif",
            "    /DISCARD/ : { *(.MIPS.abiflags .reginfo .pdr .mdebug* .options .debug* .comment .note* "
            ".gnu.attributes) }",
            "}",
            "INCLUDE undefined_syms.ld",
            "",
        ]
    )


@pytest.mark.usefixtures("per_object")
def test_file_absent_in_one_version_is_grouped_under_if(tmp_path):
    tree = make_tree(
        both("src/a.c", "f_1"),
        module("src/b.c", "main", None, us=("f_2",)),
        module("src/c.c", "main", us=("f_3",)),
        module("src/d.c", "main", eu_1=("f_4",)),
    )
    text = build.linker(make_config(tmp_path), tree).decode()
    text_pass = text.split("    {\n", 1)[1].split("    }\n", 1)[0]
    assert text_pass.count("#if defined(VERSION_US)") == 1  # b and c share one block
    assert (
        "#if defined(VERSION_EU_1)\n        . = MAX(., 0x40);\n        src/d.c.o(.text);\n"
        "        src/d.c.o(.rodata .rdata);\n#endif" in text
    )


@pytest.mark.usefixtures("per_object")
def test_conflicting_orders_fall_back_to_one_run_per_version(tmp_path):
    tree = make_tree(
        module("src/a.c", "main", None, us=("f_1",), eu_1=("f_3",)),
        module("src/b.c", "main", None, us=("f_2",), eu_1=("f_2",)),
        module("src/c.c", "main", None, us=("f_3",), eu_1=("f_1",)),
    )
    text = build.linker(make_config(tmp_path), tree).decode()
    assert (
        "#if defined(VERSION_EU_1)\n        . = MAX(., 0x10);\n        src/c.c.o(.text);\n"
        "        src/c.c.o(.rodata .rdata);\n"
        "        . = MAX(., 0x20);\n        src/b.c.o(.text);" in text
    )
    assert (
        "#if defined(VERSION_US)\n        . = MAX(., 0x10);\n        src/a.c.o(.text);\n"
        "        src/a.c.o(.rodata .rdata);\n"
        "        . = MAX(., 0x20);\n        src/b.c.o(.text);" in text
    )


def _body(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "src/" in line and ".bss" not in line]


def test_a_per_object_toolchain_keeps_literals_after_the_code_and_a_per_section_one_does_not(tmp_path, monkeypatch):
    tree = make_tree(both("src/a.c", "f_1"), both("src/b.c", "f_2"))
    rows = {
        "per-object": [
            "src/a.c.o(.text);",
            "src/a.c.o(.rodata .rdata);",
            "src/b.c.o(.text);",
            "src/b.c.o(.rodata .rdata);",
        ],
        "per-section": ["src/a.c.o(.text);", "src/b.c.o(.text);"],
    }
    for mode, want in rows.items():
        monkeypatch.setattr(build.configuration, "toolchain", lambda id, mode=mode: {"link_sections": mode})
        assert _body(build.linker(make_config(tmp_path), tree).decode()) == want, mode


@pytest.mark.usefixtures("per_object")
def test_input_sections_follow_the_rom_per_object_like_bt(tmp_path):
    """A module's data and rodata sit where the bytes are: a.data, a.rodata, then b's code, and so on."""
    tree = make_tree(
        both("src/a.c", "f_1", "d_2", "d_3", sections={"d_2": ".data", "d_3": ".rodata"}),
        both("src/b.c", "f_4", "d_5", sections={"d_5": ".data"}),
    )
    assert _body(build.linker(make_config(tmp_path), tree).decode()) == [
        "src/a.c.o(.text);",
        "src/a.c.o(.data);",
        "src/a.c.o(.rodata .rdata);",
        "src/b.c.o(.text);",
        "src/b.c.o(.rodata .rdata);",  # no rodata of its own: a per-object toolchain keeps its literals after the code
        "src/b.c.o(.data);",
    ]


@pytest.mark.usefixtures("per_object")
def test_each_object_section_is_listed_where_its_first_byte_is_even_when_the_object_is_split_over_the_rom(tmp_path):
    tree = make_tree(both("src/a.c", "f_1", "d_3", sections={"d_3": ".rodata"}), both("src/b.c", "f_2"))
    assert _body(build.linker(make_config(tmp_path), tree).decode()) == [
        "src/a.c.o(.text);",
        "src/b.c.o(.text);",
        "src/b.c.o(.rodata .rdata);",
        "src/a.c.o(.rodata .rdata);",
    ]


def test_rodata_of_code_may_live_in_a_data_segment_like_rw(tmp_path, monkeypatch):
    """Code segments list x.o(.text), the data segment lists x.o(.rodata): one rule, a section per segment."""
    monkeypatch.setattr(build.configuration, "toolchain", lambda id: {"link_sections": "per-section"})
    config = make_config(tmp_path)
    for v in ("eu-1", "us"):
        config.project.version[v].segments = (
            Segment("code", 0x1000, 0x80200400, "code"),
            Segment("rodata", 0x1800, 0x80300000, "code"),
        )
    tree = make_tree(
        both("src/a.c", "f_1", "d_160", sections={"d_160": ".rodata"}),
        both("src/b.c", "f_2", "d_161", sections={"d_161": ".rodata"}),
    )
    text = build.linker(config, tree).decode()
    code, data = text.split("    .rodata ")
    assert [line.strip() for line in code.splitlines() if "o(" in line and ".bss" not in line] == [
        "src/a.c.o(.text);",
        "src/b.c.o(.text);",
    ]
    assert [line.strip() for line in data.splitlines() if "o(" in line and ".bss" not in line] == [
        "src/a.c.o(.rodata .rdata);",
        "src/b.c.o(.rodata .rdata);",
    ]


@pytest.mark.usefixtures("no_apart")
def test_a_symbol_in_two_places_of_one_object_section_is_listed_once(tmp_path):
    tree = make_tree(both("src/a.c", "f_1", "f_2", "f_4"), both("src/b.c", "f_3"))
    body = _body(build.linker(make_config(tmp_path), tree).decode())
    assert [line for line in body if line.startswith("src/a.c.o(.text)")] == ["src/a.c.o(.text);"]


def test_splat_yaml_golden(tmp_path):
    config = make_config(tmp_path)
    text = build.splat_yaml(config, "us", make_tree()).decode()
    assert text.splitlines()[0] == "# generated from config.toml; do not edit"
    assert text.endswith(
        "\n".join(
            [
                "  compiler: KMC",
                "segments:",
                "  - name: main",
                "    type: code",
                "    start: 0x1000",
                "    vram: 0x80200400",
                "    subsegments:",
                "      - [0x1000, c, main]",
                "  - name: assets",
                "    type: bin",
                "    start: 0x2000",
                "  - [0x3000]",
                "",
            ]
        )
    )
    assert '  symbol_addrs_path: "versions/us/symbol_addrs.txt"' in text
    assert "  migrate_rodata_to_functions: true" in text
    assert "  create_c_files: false" in text
    assert '  disassemble_all: true\n  matchings_path: "nonmatchings"' in text  # every function, one folder


def test_splat_yaml_data_subsegment_per_symbol(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text(
        "f = 0x80200400; // type:func rom:0x1000\n"
        "d_2 = 0x80200600; // rom:0x1200 size:0x10\n"
        "d_1 = 0x80200500; // type:data rom:0x1100\n"
        "f_3 = 0x80200700; // type:func rom:0x1300 size:0x20\n"
        "f_4 = 0x80200740; // type:func rom:0x1340\n"
        "here = 0x80200744; // type:label rom:0x1344\n"
        "d_far = 0x80300000; // rom:0x2100\n"
    )
    held = make_tree(both("src/d.c", "d_2", sections={"d_2": ".rodata"}))
    text = build.splat_yaml(config, "us", held).decode()
    assert (
        "      - [0x1000, c, main]\n      - [0x1100, data, main/d_1]\n      - [0x1200, rodata, main/d_2]\n"
        "      - [0x1300, c, main]\n      - [0x1320, c, main]\n  - name: assets\n" in text
    )  # untyped rows are data, the tree gives the section, the code after data or a sized function restarts c
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("d_at_start = 0x80200400; // rom:0x1000\n")
    alone = build.splat_yaml(config, "us", make_tree()).decode()
    assert "      - [0x1000, data, main/d_at_start]\n  - name" in alone
    row = dict(build.configuration.toolchain("gcc-2.8.1-sn64"), data_split="symbol")
    monkeypatch.setattr(build.configuration, "toolchain", lambda id: row)
    assert "d_at_start" not in build.splat_yaml(config, "us", held).decode()  # a symbol-mode toolchain: one c row


def test_splat_yaml_writes_labels_into_the_piece_before_them(tmp_path):
    """A data name the tree calls a label (no owner, inside the data before it) starts no subsegment."""
    config = make_config(tmp_path)
    text = "d_1 = 0x80200500; // rom:0x1100\nd_2 = 0x80200504; // rom:0x1104\nd_3 = 0x80200600; // rom:0x1200\n"
    (tmp_path / "versions/us/symbol_addrs.txt").write_text(text)
    held = make_tree(both("src/d.c", "d_1", "d_3"))
    table = {"d_1": (0x80200500, 0x1100, None, "data"), "d_2": (0x80200504, 0x1104, None, "data")}
    held = Tree(held.digest, held.modules, {"us": {**table, "d_3": (0x80200600, 0x1200, None, "data")}})
    out = build.splat_yaml(config, "us", held).decode()
    assert "main/d_1]" in out and "main/d_3]" in out and "d_2" not in out


def without_hdrkey(text: str) -> str:
    """The Makefile without the C sources of hdrkey and stamprec and the stamp sweep it embeds."""
    for block in ("define HDRKEY_C\n", "define STAMPREC_C\n"):
        head, rest = text.split(block, 1)
        text = head + rest.split("\nendef", 1)[1]
    head, rest = text.split("define STAMP_SWEEP\n", 1)
    return head + rest.split("\nendef", 1)[1]


def test_makefile_variable_blocks(tmp_path):
    config = make_config(tmp_path, {"src/lib/": FileFacts(cflags=("-O1",), omit=("-O2",)), "src/a.c": FileFacts()})
    text = build.makefile(config).decode()
    assert "VMACRO_eu-1 := VERSION_EU_1" in text
    assert f"SHA1_us := {'a' * 40}" in text
    assert "ROM0_us := 4096" in text
    assert "ROM_SECTIONS_us :=\n" in text
    block = "$(foreach v,$(VERSIONS),build/$(v)/src/lib/%.o)"
    lines = [line for line in text.splitlines() if line.startswith(block)]
    assert [line.split(": ")[1].split(" ")[0] for line in lines] == ["TC", "CPPFLAGS", "CFLAGS"]
    assert lines[2].endswith("-mno-fix4300 -O1") and " -O2 " not in lines[2]
    assert "$(foreach v,$(VERSIONS),build/$(v)/src/a.c.o): TC := gcc_2_8_1_sn64" in text
    make_text = without_hdrkey(text)
    assert "sed -nE" in text
    assert sum(line.count("sed ") for line in make_text.splitlines() if not line.startswith("#")) == 4
    # PROVIDE lines (facts, .L labels); data files; code files; jump tables
    assert "DATA_SED := -e '/^\\.section /d'" in text and "s/^\\.align [4-9]/.align 3/" in text
    assert "PATCHES" not in text and " dd " not in text


def test_a_file_rows_preprocessor_flags_reach_cpp_not_the_compiler(tmp_path):
    config = make_config(tmp_path, {"src/a.c": FileFacts(cflags=("-UF3DEX_GBI_2", "-D", "X=1", "-G8"))})
    lines = [line for line in build.makefile(config).decode().splitlines() if "build/$(v)/src/a.c.o)" in line]
    assert lines[1].endswith("-DF3DEX_GBI_2 -UF3DEX_GBI_2 -D X=1")
    assert lines[2].endswith(" -G8") and "-U" not in lines[2] and "X=1" not in lines[2]


def line_count(tmp_path: Path, n: int) -> int:
    """The Makefile of a project whose tree holds n modules: it never reads the tree."""
    make_tree(*(both(f"src/f{i}.c", f"f_{i}") for i in range(n)))
    return len(build.makefile(make_config(tmp_path)).decode().splitlines())


def test_makefile_has_no_per_file_rules(tmp_path):
    assert line_count(tmp_path, 3) == line_count(tmp_path, 300)
    text = build.makefile(make_config(tmp_path)).decode()
    assert len(without_hdrkey(text).splitlines()) < 520
    assert "src/f1.c" not in text


def test_link_script_grows_with_files_but_makefile_does_not(tmp_path):
    small = build.linker(make_config(tmp_path), make_tree(*(both(f"src/f{i}.c", f"f_{i}") for i in range(3))))
    big = build.linker(make_config(tmp_path), make_tree(*(both(f"src/f{i}.c", f"f_{i}") for i in range(300))))
    assert len(big.splitlines()) > len(small.splitlines())


def test_local_mk_names_pinned_tools_only_by_host_path(tmp_path, monkeypatch):
    host = SimpleNamespace(workers=4, tools={"cpp": "/usr/bin/cpp"}, toolchain_root=Path("/chains"))
    text = build.local_mk(SimpleNamespace(host=host)).decode()
    assert "JOBS := 4" in text
    assert "CPP := /usr/bin/cpp" in text
    assert "ABUMAS :=" not in text  # a pinned tool is the Makefile's own fetch


def test_toolchains_install_in_one_pool_dispatch(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(
        build.pool,
        "map",
        lambda config, name, function, items, key=None: calls.append((name, items)) or [Path(i[1]) for i in items],
    )
    assert build.install_toolchains(cfg, ["x", "y"]) == [Path("x"), Path("y")]
    assert calls == [("build.install_toolchains", [(cfg, "x"), (cfg, "y")])]


def _archive(tmp_path: Path, files: dict[str, bytes]) -> Path:
    import io
    import tarfile

    path = tmp_path / "chain.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _chain_row(tmp_path: Path, files: dict[str, bytes], pins: dict[str, str]) -> dict:
    from unbake.contracts import sha256_file

    archive = _archive(tmp_path, files)
    return {
        "pins": pins,
        "downloads": [
            {"url": "https://example.invalid/chain.tar.gz", "sha256": sha256_file(archive), "files": list(files)}
        ],
    }


def test_install_toolchain_fetches_unpacks_marks_executable_and_checks_the_pins(tmp_path, monkeypatch):
    from unbake.contracts import sha256

    files = {"cc": b"compiler", "as": b"assembler"}
    row = _chain_row(tmp_path, files, {n: sha256(d) for n, d in files.items()})
    monkeypatch.setattr(build.configuration, "toolchains", lambda: {"chain": row})
    fetched = []
    archive = _archive(tmp_path, files)
    monkeypatch.setattr(build, "_fetch", lambda config, download: fetched.append(download["url"]) or archive)
    config = SimpleNamespace(host=SimpleNamespace(toolchain_root=tmp_path / "chains"))
    dest = build.install_toolchain(config, "chain")
    assert dest == tmp_path / "chains/chain" and (dest / "cc").read_bytes() == b"compiler"
    assert (dest / "cc").stat().st_mode & 0o111 and fetched == ["https://example.invalid/chain.tar.gz"]
    assert build.install_toolchain(config, "chain") == dest and len(fetched) == 1  # pinned and present: no second fetch


def test_install_toolchain_refuses_what_the_pins_or_the_archive_do_not_hold(tmp_path, monkeypatch):
    from unbake.contracts import Refusal

    config = SimpleNamespace(host=SimpleNamespace(toolchain_root=tmp_path / "chains"))
    monkeypatch.setattr(build.configuration, "toolchains", lambda: {})
    with pytest.raises(Refusal) as unknown:
        build.install_toolchain(config, "chain")
    assert unknown.value.findings[0].key == "toolchain.pin"
    row = _chain_row(tmp_path, {"cc": b"compiler"}, {"cc": "0" * 64})
    monkeypatch.setattr(build.configuration, "toolchains", lambda: {"chain": row})
    monkeypatch.setattr(build, "_fetch", lambda config, download: _archive(tmp_path, {"cc": b"compiler"}))
    with pytest.raises(Refusal) as pinned:
        build.install_toolchain(config, "chain")
    assert pinned.value.findings[0].key == "toolchain.pin" and pinned.value.findings[0].missing == ("cc",)
    other = {**row, "pins": {"cc": "1" * 64}}
    other["downloads"] = [{**row["downloads"][0], "files": ["cc", "absent"]}]
    monkeypatch.setattr(build.configuration, "toolchains", lambda: {"chain2": other})
    with pytest.raises(Refusal) as short:
        build.install_toolchain(config, "chain2")
    assert short.value.findings[0].key == "toolchain.download" and short.value.findings[0].missing == ("absent",)


def test_a_download_is_checked_against_its_sha256_and_kept(tmp_path, monkeypatch):
    import io

    from unbake.contracts import Refusal, sha256

    config = SimpleNamespace(host=SimpleNamespace(toolchain_root=tmp_path))
    data = b"archive"
    row = {"url": "https://example.invalid/a.tar.gz", "sha256": sha256(data)}
    opened = []
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url: opened.append(url) or io.BytesIO(data))
    path = build._fetch(config, row)
    assert path.read_bytes() == data and build._fetch(config, row) == path and len(opened) == 1
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url: io.BytesIO(b"other"))
    with pytest.raises(Refusal) as refused:
        build._fetch(config, {**row, "sha256": "0" * 64, "url": "https://example.invalid/b.tar.gz"})
    assert refused.value.findings[0].key == "toolchain.download"


def test_a_clean_make_check_records_the_exact_versions_of_the_tree(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.host = SimpleNamespace(slots=2, workers=2)
    monkeypatch.setattr(build, "_make", lambda cfg, target: SimpleNamespace(exit=0, stderr=b""))
    held = make_tree(both("src/a.c", "f_1"))
    assert build.make_check(config, held) == {"exit": 0, "versions": ["eu-1", "us"]}
    document = json.loads((tmp_path / ".unbake/check.json").read_text())
    assert (document["digest"], document["exact"], document["failing"]) == ("digest", ["eu-1", "us"], [])
    build.make_check(config, held)  # the inputs have not moved: the last check still holds, nothing is run
    monkeypatch.setattr(build, "_make", Mock(side_effect=AssertionError("make must not run again")))
    assert build.make_check(config, held)["exit"] == 0


def test_a_failed_make_check_revokes_credit_and_names_every_failed_target(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    failure = SimpleNamespace(
        exit=2, stderr=b"make: *** [build/us/Game.z64] Error 1\nmake: *** [build/eu-1/Game.z64] Error 1\n"
    )
    monkeypatch.setattr(build, "_make", lambda cfg, target: failure)
    with pytest.raises(Refusal) as refused:
        build.make_check(config, make_tree())
    assert refused.value.findings[0].key == "setup.make" and "build/us/Game.z64" in refused.value.findings[0].reason
    assert json.loads((tmp_path / ".unbake/check.json").read_bytes())["exact"] == []


def test_text_sizes_give_each_object_its_code_up_to_the_next_owned_bytes(tmp_path):
    """make_tree spaces names 0x10 apart: a's code runs to its data, b's to c's code; an object split over the ROM
    and the segment's last object (its end needs the next segment) get no row."""
    tree = make_tree(
        both("src/a.c", "f_1", "f_2", "d_3"),
        both("src/b.c", "f_4"),
        both("src/c.c", "f_5", "f_7"),
        both("src/d.c", "f_6"),
        both("src/e.s", "f_8"),
    )
    assert build.text_sizes(make_config(tmp_path), tree, "us").decode().splitlines()[1:] == [
        "build/us/src/a.c.o: TEXTSIZE := 0x20",
        "build/us/src/b.c.o: TEXTSIZE := 0x10",
        "build/us/src/d.c.o: TEXTSIZE := 0x10",
        "build/us/src/e.s.o: TEXTSIZE := 0xF80",  # the code segment ends where the next one starts
        # d's code lies between c's two functions: c is placed function by function
        "build/us/src/c.c.o: FSPLIT := fsplit",
    ]
    assert "src/e.s" not in build.text_sizes(make_config(tmp_path), tree, "eu-1").decode()  # the last segment


def test_the_makefile_cuts_a_zero_tail_of_text_and_spells_what_the_assembler_would_change(tmp_path):
    text = build.makefile(make_config(tmp_path)).decode()
    assert "-include $(VERSIONS:%=versions/%/text_sizes.mk)" in text
    assert text.count("$(Q)$(if $(TEXTSIZE),$(TRIM_TEXT) $@ $(TEXTSIZE))") == 1  # assembled objects
    assert "$(call shq,$(TEXTSIZE))" in text  # compiled ones: the object helper trims them when they compile
    assert "(ld|sd|ldl|ldr|sdl|sdr|lld|scd|ddiv|ddivu)" in text and "1000[0-9A-F]{4}" in text


def test_trim_text_cuts_only_a_zero_tail(tmp_path):
    """cc_cache's trim rewrites the .text size of a big-endian ELF32 when the bytes past the size are zero."""
    import struct

    def elf(text: bytes) -> bytes:
        names = b"\0.text\0.shstrtab\0"
        body = text + names
        shoff = 52 + len(body)
        head = (
            b"\x7fELF\x01\x02\x01"
            + bytes(9)
            + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, shoff, 0, 52, 0, 0, 40, 3, 2)
        )
        null = bytes(40)
        text_h = struct.pack(">IIIIIIIIII", 1, 1, 6, 0, 52, len(text), 0, 0, 16, 0)
        str_h = struct.pack(">IIIIIIIIII", 7, 3, 0, 0, 52 + len(text), len(names), 0, 0, 1, 0)
        return head + body + null + text_h + str_h

    def size_after(text: bytes, want: int) -> int:
        path = tmp_path / "x.o"
        path.write_bytes(elf(text))
        cc_cache.trim_text(str(path), hex(want))
        data = path.read_bytes()
        shoff = struct.unpack_from(">I", data, 32)[0]
        return struct.unpack_from(">I", data, shoff + 40 + 20)[0]

    assert size_after(b"\x03\xe0\x00\x08" + bytes(12), 8) == 8
    assert size_after(b"\x03\xe0\x00\x08" + bytes(4) + b"\x01" + bytes(3), 8) == 12  # not padding: left as it is
    assert size_after(b"\x03\xe0\x00\x08" + bytes(4), 16) == 8  # never grown


def test_splat_yaml_starts_no_data_piece_off_a_word_boundary(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text(
        "d_1 = 0x80200500; // rom:0x1100\nd_2 = 0x80200502; // rom:0x1102\nd_3 = 0x80200504; // rom:0x1104\n"
    )
    text = build.splat_yaml(config, "us", make_tree()).decode()
    assert "main/d_1]" in text and "main/d_3]" in text and "d_2" not in text


@pytest.mark.parametrize("suffix", [".c", ".s"])
@pytest.mark.parametrize("before,after", [("0x10", "0x14"), ("0x10", None), (None, "0x10")])
def test_text_size_changes_rebuild_only_the_affected_object(tmp_path, suffix, before, after):
    """Exercise the generated make rules with a mocked compiler/assembler and trim operation."""
    import subprocess
    import sys

    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "src").mkdir()
    (tmp_path / "asm/us").mkdir(parents=True)
    (tmp_path / "asm/us/.extracted").touch()
    targets = [f"build/us/src/{name}{suffix}.o" for name in ("a", "b")]
    for name in ("a", "b"):
        (tmp_path / f"src/{name}{suffix}").write_text(name)
    compiler = tmp_path / "compiler.py"
    compiler.write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[-1]).write_bytes(Path(sys.argv[-3]).read_bytes())\n"
        "with Path('compiled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
    )

    def sizes(size):
        rows = [f"{targets[1]}: TEXTSIZE := 0x20"]
        if size is not None:
            rows.append(f"{targets[0]}: TEXTSIZE := {size}")
        (tmp_path / "versions/us/text_sizes.mk").write_text("\n".join(rows) + "\n")

    def make():
        (tmp_path / "compiled").write_text("")
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-j1",
                *targets,
                "STAMPS=",
                "TC=stub",
                f"AS={sys.executable} {compiler}",
                _pp("PREPROCESS_stub", "cat $<"),
                "COMPILE_stub=$(AS) $(@:.o=.i) -o $@",
                "TRIM_TEXT=true",
                CCPY,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return sorted((tmp_path / "compiled").read_text().splitlines())

    sizes(before)
    assert make() == targets
    assert make() == []
    sizes(after)
    assert make() == targets[:1]
    assert make() == []
    sizes(before)
    assert make() == targets[:1]  # restoring an older value must rebuild too


def test_an_object_compiles_again_only_when_its_preprocessed_text_changes(tmp_path):
    """The generated rules with a stub preprocessor that drops comment lines: a comment edit preprocesses the source
    again and compiles nothing; a code edit compiles that one object; nothing changed runs nothing."""
    import subprocess
    import sys

    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "src").mkdir()
    (tmp_path / "asm/us").mkdir(parents=True)
    (tmp_path / "asm/us/.extracted").touch()
    (tmp_path / "versions/us/text_sizes.mk").write_text("")
    pre = tmp_path / "pre.py"
    pre.write_text(
        "from pathlib import Path\nimport sys\n"
        "text = Path(sys.argv[1]).read_text().splitlines(keepends=True)\n"
        "sys.stdout.write(''.join(t for t in text if not t.startswith('//')))\n"
        "with Path('preprocessed').open('a') as log: log.write(sys.argv[1] + '\\n')\n"
    )
    compiler = tmp_path / "compiler.py"
    compiler.write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[-1]).write_bytes(Path(sys.argv[-3]).read_bytes())\n"
        "with Path('compiled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
    )
    targets = ["build/us/src/a.c.o", "build/us/src/b.c.o"]

    def make():
        for log in ("compiled", "preprocessed"):
            (tmp_path / log).write_text("")
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-j1",
                *targets,
                "STAMPS=",
                "TC=stub",
                f"AS={sys.executable} {compiler}",
                _pp("PREPROCESS_stub", f"{sys.executable} {pre} $<"),
                "COMPILE_stub=$(AS) $(@:.o=.i) -o $@",
                "TRIM_TEXT=true",
                CCPY,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return (
            sorted((tmp_path / "preprocessed").read_text().splitlines()),
            sorted((tmp_path / "compiled").read_text().splitlines()),
        )

    for name in ("a", "b"):
        (tmp_path / f"src/{name}.c").write_text(f"// note\nint {name};\n")
    assert make() == (["src/a.c", "src/b.c"], targets)
    assert make() == ([], [])

    def edit(text):  # everything built ages first, so the edit is newer than every output and older than the next
        now = time.time_ns()
        for path in tmp_path.rglob("*"):
            age = 10**10 if "build" in path.relative_to(tmp_path).parts else 2 * 10**10
            age += 10**9 if path.suffix == ".i" else 0  # each text older than the object made from it
            os.utime(path, ns=(now - age, now - age))
        (tmp_path / "src/a.c").write_text(text)
        os.utime(tmp_path / "src/a.c", ns=(now - 5 * 10**9, now - 5 * 10**9))

    edit("// another note\nint a;\n")
    assert make() == (["src/a.c"], [])  # the text is the same: nothing compiles
    assert make() == ([], [])
    edit("// another note\nint a2;\n")
    assert make() == (["src/a.c"], targets[:1])
    (tmp_path / "build/us/src/b.c.i.ok").unlink()  # objects built before the stamps: nothing newer, nothing to do
    assert make() == ([], [])


def test_an_extraction_that_changes_nothing_keeps_every_date(tmp_path):
    """splat writes every file again. Unchanged bytes keep their dates; a changed file takes a new date but the stamp
    (every object's input) keeps its own, as each object naming the file follows it (asm.deps); an object that named a
    file the extraction no longer makes is dated long ago; a change under the shared include folder moves the stamp."""
    import subprocess
    import sys

    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "roms").mkdir(exist_ok=True)
    (tmp_path / "roms/baserom.us.z64").write_bytes(bytes(64))
    (tmp_path / "versions/us/splat.yaml").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text('INCLUDE_ASM("seg", g);\n')
    (tmp_path / "out.txt").write_text("call f\n")
    (tmp_path / "names.txt").write_text("f g")
    (tmp_path / "macro.txt").write_text("macro\n")
    splat = tmp_path / "splat.py"
    splat.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "if '--version' in sys.argv:\n"
        "    print('splat test')\n"
        "    sys.exit(0)\n"
        "folder = Path('asm/us/nonmatchings/seg')\n"
        "folder.mkdir(parents=True, exist_ok=True)\n"
        "for old in folder.glob('*.s'):\n"
        "    old.unlink()\n"
        "for name in Path('names.txt').read_text().split():\n"
        "    (folder / (name + '.s')).write_text(Path('out.txt').read_text() + name + '\\n')\n"
        "Path('asm/us/include').mkdir(exist_ok=True)\n"
        "Path('asm/us/include/macro.inc').write_text(Path('macro.txt').read_text())\n"
    )
    files = ("asm/us/nonmatchings/seg/f.s", "asm/us/nonmatchings/seg/g.s", "asm/us/include/macro.inc")
    stamp = "asm/us/.extracted"

    def extract():
        result = subprocess.run(
            ["make", "--no-print-directory", "extract-us", f"SPLAT={sys.executable} {splat}"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return {p: (tmp_path / p).stat().st_mtime_ns for p in (stamp, *files) if (tmp_path / p).exists()}

    def age(*paths):
        for path in paths:
            os.utime(tmp_path / path, ns=(old, old))

    first = extract()
    old = first[stamp] - 10**10
    age(*first)
    assert extract() == dict.fromkeys(first, old)
    assert not list((tmp_path / "build").glob(".extract-dates-*"))
    (tmp_path / "out.txt").write_text("call f_renamed\n")
    (tmp_path / "versions/us/splat.yaml").write_text("# renamed\n")
    changed = extract()
    assert changed[files[0]] > old and changed[files[1]] > old and changed[files[2]] == old
    assert changed[stamp] == old
    age(*changed)
    obj = tmp_path / "build/us/src/a.c.o"
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_text("o")
    (tmp_path / "names.txt").write_text("f")  # g.s, which a.c names, is no longer extracted
    (tmp_path / "versions/us/splat.yaml").write_text("# 2\n")
    gone = extract()
    assert gone[stamp] == old and files[1] not in gone
    assert obj.stat().st_mtime_ns == build._LONG_AGO
    age(*gone)
    (tmp_path / "macro.txt").write_text("macro 2\n")  # every object assembles the shared include folder
    (tmp_path / "versions/us/splat.yaml").write_text("# 3\n")
    assert extract()[stamp] > old


def test_an_object_follows_the_extracted_files_its_include_lines_name(tmp_path):
    """The compiler's assembly holds only an `.include` of each INCLUDE_ASM/RODATA/DATA file; the assembler reads its
    content. A stub compiler appends the named files' text to the object, as the assembler does: a file edited in place
    (a symbol renamed, the source and its .i unchanged) must rebuild exactly the objects that name it."""
    import subprocess
    import sys

    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "src").mkdir()
    (tmp_path / "asm/us/nonmatchings/seg").mkdir(parents=True)
    (tmp_path / "asm/us/data/seg").mkdir(parents=True)
    (tmp_path / "asm/eu-1/nonmatchings/seg").mkdir(parents=True)
    (tmp_path / "asm/us/.extracted").touch()
    (tmp_path / "versions/us/text_sizes.mk").write_text("")
    (tmp_path / "src/a.c").write_text(
        'INCLUDE_ASM("seg", f_a);\nINCLUDE_RODATA("seg", d_a);\nINCLUDE_ASM("seg", f_eu);\n'
    )
    (tmp_path / "src/b.c").write_text('INCLUDE_ASM("seg", f_b);\nINCLUDE_ASM("seg", f_gone);\n')
    (tmp_path / "src/c.c").write_text("int c;\n")
    files = {
        "asm/us/nonmatchings/seg/f_a.s": "call old_name\n",
        "asm/us/data/seg/d_a.s": "data old_name\n",
        "asm/eu-1/nonmatchings/seg/f_eu.s": "only the other version has it\n",
        "asm/us/nonmatchings/seg/f_b.s": "b body\n",
    }
    for path, text in files.items():
        (tmp_path / path).write_text(text)
    compiler = tmp_path / "compiler.py"
    compiler.write_text(
        "from pathlib import Path\nimport re, sys\n"
        "text = Path(sys.argv[-3]).read_text()\n"
        "for kind, seg, name in re.findall(r'INCLUDE_(ASM|RODATA)\\(\"(\\w+)\", (\\w+)\\)', text):\n"
        "    where = Path('asm/us') / ('nonmatchings' if kind == 'ASM' else 'data') / seg / (name + '.s')\n"
        "    text += where.read_text() if where.exists() else ''\n"
        "Path(sys.argv[-1]).write_text(text)\n"
        "with Path('compiled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
    )
    targets = [f"build/us/src/{name}.c.o" for name in "abc"]

    def make():
        (tmp_path / "compiled").write_text("")
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-j1",
                *targets,
                "STAMPS=",
                "TC=stub",
                f"AS={sys.executable} {compiler}",
                _pp("PREPROCESS_stub", "cat $<"),
                "COMPILE_stub=$(AS) $(@:.o=.i) -o $@",
                "TRIM_TEXT=true",
                CCPY,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return sorted((tmp_path / "compiled").read_text().splitlines())

    def edit(path, text):  # everything built ages first, so the edit is newer than every output
        now = time.time_ns()
        for old in tmp_path.rglob("*"):
            os.utime(old, ns=(now - 2 * 10**10, now - 2 * 10**10))
        (tmp_path / path).write_text(text)

    assert make() == targets
    assert make() == []
    assert "old_name" in (tmp_path / targets[0]).read_text()
    edit("asm/us/nonmatchings/seg/f_a.s", "call new_name\n")  # the .i of a.c is the same text
    assert make() == targets[:1]
    assert "new_name" in (tmp_path / targets[0]).read_text()
    assert make() == []
    edit("asm/us/data/seg/d_a.s", "data new_name\n")  # an INCLUDE_RODATA file follows too
    assert make() == targets[:1]
    edit("asm/us/nonmatchings/seg/f_b.s", "b changed\n")  # only b names it
    assert make() == targets[1:2]
    edit("asm/eu-1/nonmatchings/seg/f_eu.s", "another version\n")  # a file of a version not built: nothing
    assert make() == []


@pytest.mark.parametrize("mode", ["symbol", "subsegment"])
@pytest.mark.parametrize("offset,kind", [(0, "data"), (1, "data"), (2, "data"), (3, "data"), (1, "func"), (1, "label")])
def test_folded_data_symbols_have_a_finding_and_report_line(tmp_path, monkeypatch, mode, offset, kind):
    from contextlib import nullcontext

    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text(
        "base = 0x80200400; // type:data rom:0x1000\n"
        f"tail = 0x{0x80200410 + offset:X}; // type:{kind} rom:0x{0x1010 + offset:X}\n"
    )
    row = dict(build.configuration.toolchain("gcc-2.8.1-sn64"), data_split=mode)
    monkeypatch.setattr(build.configuration, "toolchain", lambda id: row)
    found, said = [], []
    span = SimpleNamespace(add=lambda **values: found.extend(values.get("findings", [])))
    monkeypatch.setattr(build.effort, "stage", lambda name: nullcontext(span))
    monkeypatch.setattr(build.effort, "say", said.append)
    text = build.splat_yaml(config, "us", make_tree()).decode()
    folded = offset != 0 and kind == "data"
    assert len(found) == int(folded)
    assert said == [f.reason for f in found]
    if folded:
        finding = found[0]
        assert (finding.key, finding.unit, finding.path, finding.versions, finding.blocking) == (
            "symbol.folded",
            "tail",
            "versions/us/symbol_addrs.txt",
            ("us",),
            False,
        )
        assert "tail" in finding.reason and f"0x{0x80200410 + offset:08X}" in finding.reason
        assert "main/tail" not in text


def test_code_sed_keeps_the_relocation_of_a_64_bit_load_and_words_the_rest(tmp_path):
    """CODE_SED of the generated Makefile, run by make and sed on extracted lines."""
    import subprocess

    lines = build.makefile(make_config(tmp_path)).decode().splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("CODE_SED :="))
    end = next(i for i in range(at, len(lines)) if not lines[i].endswith("\\"))
    (tmp_path / "rules.mk").write_text("\n".join(lines[at : end + 1]) + "\nrun:\n\t@sed -E $(CODE_SED) in.s\n")
    (tmp_path / "in.s").write_text(
        "/* 1234 80001234 DC820010 */  ld          $v0, %lo(D_80001000)($a0)\n"
        "/* 1238 80001238 DC820008 */  ld          $v0, 0x8($a0)\n"
        "/* 123C 8000123C 24020001 */  addiu       $v0, $zero, 0x1\n"
    )
    out = subprocess.run(["make", "-s", "-f", "rules.mk", "run"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert out.splitlines() == [
        "/* 1234 80001234 DC820010 */  .reloc ., R_MIPS_LO16, D_80001000",
        "    .word 0xDC820000",
        "/* 1238 80001238 DC820008 */  .word 0xDC820008",
        "/* 123C 8000123C 24020001 */  addiu       $v0, $zero, 0x1",
    ]


def test_code_sed_words_a_handwritten_line_but_keeps_one_that_carries_a_relocation(tmp_path):
    """A handwritten lui/lw pair on a symbol keeps %hi/%lo for the link; one with no symbol is its word."""
    import subprocess

    lines = build.makefile(make_config(tmp_path)).decode().splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("CODE_SED :="))
    end = next(i for i in range(at, len(lines)) if not lines[i].endswith("\\"))
    (tmp_path / "rules.mk").write_text("\n".join(lines[at : end + 1]) + "\nrun:\n\t@sed -E $(CODE_SED) in.s\n")
    mark = "/* handwritten instruction */"
    (tmp_path / "in.s").write_text(
        f"/* 1 80001000 3C0C800D */  lui         $t4, %hi(D_800D92A4) {mark}\n"
        f"/* 5 80001004 8D8C92A4 */  lw          $t4, %lo(D_800D92A4)($t4) {mark}\n"
        f"/* 9 80001008 40026000 */  mfc0        $v0, $12 {mark}\n"
    )
    out = subprocess.run(["make", "-s", "-f", "rules.mk", "run"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert out.splitlines() == [
        f"/* 1 80001000 3C0C800D */  lui         $t4, %hi(D_800D92A4) {mark}",
        f"/* 5 80001004 8D8C92A4 */  lw          $t4, %lo(D_800D92A4)($t4) {mark}",
        "/* 9 80001008 40026000 */  .word 0x40026000",
    ]


def test_extract_rewrites_each_data_piece_as_its_rom_slice_keeping_labels(tmp_path):
    """FILL_DATA of the generated Makefile: splat's short data file becomes the ROM bytes up to the next row."""
    import subprocess
    import sys

    text = build.makefile(make_config(tmp_path)).decode()
    code = text.split("define FILL_DATA\n", 1)[1].split("\nendef", 1)[0]
    work = tmp_path / "work"
    (work / "roms").mkdir(parents=True)
    (work / "roms/baserom.us.z64").write_bytes(bytes(range(0x40)) * 0x100)
    (work / "versions/us").mkdir(parents=True)
    (work / "versions/us/splat.yaml").write_text(
        "segments:\n  - name: main\n    start: 0x1000\n    subsegments:\n      - [0x1000, c, main]\n"
        "      - [0x1010, data, main/d_1]\n      - [0x1016, c, main]\n  - [0x2000]\n"
    )
    (work / "asm/us/data/main").mkdir(parents=True)
    (work / "asm/us/data/main/d_1.s").write_text(
        '.include "macro.inc"\n\ndlabel d_1\n/* 1010 80200410 10111213 */ .word 0x10111213\ndlabel d_2\n'
        "/* 1014 80200414 1415 */ .short 0x1415\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code, "us"], cwd=work, check=True)
    assert (work / "asm/us/data/main/d_1.s").read_text().splitlines() == [
        '.include "macro.inc"',
        "dlabel d_1",
        "    .byte 0x10, 0x11, 0x12, 0x13",
        "dlabel d_2",
        "    .byte 0x14, 0x15",
    ]


def test_an_object_whose_data_starts_off_a_word_gets_noalign_and_its_alignment_dropped(tmp_path):
    """text_sizes.mk marks the object (tree facts); STRIP_ALIGN of the generated Makefile drops data alignment."""
    import subprocess
    import sys

    held = make_tree(both("src/a.c", "f_1", "d_2"), both("src/b.c", "d_3"))
    table = {v: dict(rows) for v, rows in held.symbols.items()}
    table["us"]["d_3"] = (0x80200032, 0x1032, 0x10, "data")
    held = Tree(held.digest, held.modules, table)
    config = make_config(tmp_path)
    rows = build.text_sizes(config, held, "us").decode().splitlines()
    assert "build/us/src/b.c.o: NOALIGN := 1" in rows and not any("a.c.o: NOALIGN" in r for r in rows)
    text = build.makefile(config).decode()
    assert "$(if $(NOALIGN),'&&' @py STRIP_ALIGN $(@:.o=.s) $(NOALIGN))" in text
    code = text.split("define STRIP_ALIGN\n", 1)[1].split("\nendef", 1)[0]
    source = tmp_path / "x.s"
    source.write_text(".text\n.align 2\nf:\n.section .rodata\n.align 3\n.word 1\n.half 2\n.text\n.align 2\n")
    subprocess.run([sys.executable, "-I", "-c", code, str(source), "2"], check=True)
    assert source.read_text() == ".text\n.align 2\nf:\n.section .rodata\n.4byte 1\n.2byte 2\n.text\n.align 2\n"


def test_data_starting_word_aligned_but_off_eight_loses_only_the_eight_alignment(tmp_path):
    """A unit's .rodata at 4 mod 8 (a jump table behind `.align 3`): the ROM pads by address, gas by section start."""
    import subprocess
    import sys

    held = make_tree(both("src/a.c", "f_1", "d_2"), both("src/b.c", "d_3"))
    table = {v: dict(rows) for v, rows in held.symbols.items()}
    table["us"]["d_3"] = (0x80200034, 0x1034, 0x10, "data")
    held = Tree(held.digest, held.modules, table)
    config = make_config(tmp_path)
    assert not any(
        "NOALIGN" in r for r in build.text_sizes(config, held, "us").decode().splitlines()
    )  # the Makefile keeps its own
    text = build.makefile(config).decode()
    code = text.split("define STRIP_ALIGN\n", 1)[1].split("\nendef", 1)[0]
    source = tmp_path / "x.s"
    source.write_text(".text\n.align 3\nf:\n.section .rodata\n.word 1\n.align 2\n.align 3\n.word 2\n.half 3\n")
    subprocess.run([sys.executable, "-I", "-c", code, str(source), "4"], check=True)
    assert source.read_text() == ".text\n.align 3\nf:\n.section .rodata\n.word 1\n.align 2\n.word 2\n.half 3\n"


def test_a_toolchain_row_biases_jump_tables_and_allocates_rdata():
    """jump_table_base and rdata_alloc of a toolchain row become recipe lines around its assembler."""
    lines = build._recipe("gcc-2.8.1-sn64").splitlines()[-1].split(" '&&' ")
    table = next(i for i, line in enumerate(lines) if line.startswith("@jumptable"))
    asm = max(i for i, line in enumerate(lines) if "asn64" in line)  # the plain assembler, after the apart branch's
    assert table < asm and lines[table] == "@jumptable $(@:.o=.s) 80000000"
    assert lines[asm + 1] == "$(OBJCOPY) --set-section-flags .rdata=alloc,load,readonly,data,contents $@"
    assert "OBJCOPY" not in build._recipe("gcc-2.7.2-kmc") and "@jumptable" not in build._recipe("gcc-2.7.2-kmc")


def test_a_file_row_pads_hilo_hazards_the_way_the_rom_assembler_did(tmp_path):
    """A file row's hilo_nops: two nops between an mfhi/mflo and an adjacent mult/div (gas pads only the gap of one)."""
    import subprocess
    import sys

    recipe = build._recipe("gcc-2.8.1-sn64").splitlines()[-1]
    assert "$(if $(HILO),'&&' @py HILO_HAZARD $(@:.o=.s))" in recipe
    assert recipe.index("HILO_HAZARD") < recipe.index("asn64")
    code = build.makefile(make_config(tmp_path)).decode().split("define HILO_HAZARD\n", 1)[1].split("\nendef", 1)[0]
    source = tmp_path / "x.s"
    source.write_text(
        "f:\n\tmfhi\t$3\n\tmult\t$4,$5\n\tmflo\t$6\n\tandi\t$2,$3,1\n\tdivu\t$4,$5\n"  # adjacent; gap of one
        "\t.set\tnoreorder\n\tmflo\t$6\n\tmult\t$4,$5\n\t.set\treorder\n"  # noreorder: untouched
        "\tmfhi\t$3 # c\n$L1:\n\t#nop\n\tmthi\t$4\n\tmfhi\t$3\n\tdivu\t$2,$4,$5\n"  # label ignored; 3-operand div
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(source)], check=True)
    assert source.read_text() == (
        "f:\n\tmfhi\t$3\n\tnop\n\tnop\n\tmult\t$4,$5\n\tmflo\t$6\n\tandi\t$2,$3,1\n\tdivu\t$4,$5\n"
        "\t.set\tnoreorder\n\tmflo\t$6\n\tmult\t$4,$5\n\t.set\treorder\n"
        "\tmfhi\t$3 # c\n$L1:\n\t#nop\n\tnop\n\tnop\n\tmthi\t$4\n\tmfhi\t$3\n\tdivu\t$2,$4,$5\n"
    )


def test_undefined_syms_provide_the_aliases_too(tmp_path):
    """make build/us/undefined_syms.ld PROVIDEs aliases.txt rows next to symbol_addrs.txt ones."""
    import subprocess

    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("f = 0x80200400; // type:func rom:0x1000\n")
    (tmp_path / "versions/us/aliases.txt").write_text("g = 0x80200400;\n")
    (tmp_path / "build/us").mkdir(parents=True)
    for name in ("undefined_syms_auto.txt", "undefined_funcs_auto.txt"):
        (tmp_path / "build/us" / name).write_text("")
    (tmp_path / "asm/us").mkdir(parents=True)
    (tmp_path / "asm/us/.extracted").write_text("")
    (tmp_path / "build/us/caller.CPP.i").write_text("extern short D_auto_800CAB8E[];\n")
    subprocess.run(["make", "-s", "build/us/undefined_syms.ld"], cwd=tmp_path, check=True, capture_output=True)
    rows = (tmp_path / "build/us/undefined_syms.ld").read_text().splitlines()
    assert "PROVIDE(f = 0x80200400);" in rows and "PROVIDE(g = 0x80200400);" in rows
    assert "PROVIDE(D_auto_800CAB8E = 0x800CAB8E);" in rows


def test_undefined_syms_waits_for_every_object_it_reads_and_reads_no_object(tmp_path):
    """The rule that writes undefined_syms.ld depends on every object of the version (the .i and .s it greps are
    complete once the object is built) and feeds its sed the fact files only, never a binary object."""
    text = build.makefile(make_config(tmp_path)).decode()
    rule = text[text.index("build/%/undefined_syms.ld:") :].split("\n", 2)
    assert "asm/%/.extracted" in rule[0] and "OBJS_" not in rule[0]  # make 4.4.1 segfaults on it as a second expansion
    assert "eval build/$(v)/undefined_syms.ld: $(OBJS_$(v))" in text
    assert "$(filter-out %.o,$^) > $@" in rule[1]


def test_segments_sharing_a_name_are_pieces_of_one_tag(tmp_path):
    """Rows of one version may share a name (one data segment the link cuts around a blob): splat and the link name
    each piece apart, a later one `<name>_<ROM>`, its asm keeps the shared folder, and each lists only its own range."""
    config = make_config(tmp_path)
    config.project.version["us"].segments = (
        Segment("main", 0x1000, 0x80200400, "code"),
        Segment("blob", 0x1800, None, "bin"),
        Segment("main", 0x1900, 0x80200D00, "code"),
        Segment("assets", 0x2000, None, "bin"),
    )
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("d_150 = 0x80200D60; // rom:0x1960\n")
    tree = make_tree(module("src/a.c", "main", us=("f_1",)), module("src/b.c", "main", us=("d_150",)))
    yaml = build.splat_yaml(config, "us", tree).decode()
    assert [line for line in yaml.splitlines() if line.startswith("  - name:")] == [
        "  - name: main",
        "  - name: blob",
        "  - name: main_1900",
        "  - name: assets",
    ]
    assert "      - [0x1960, data, main/d_150]" in yaml
    text = build.linker(config, tree).decode()
    first, second = text.split("    .main_1900 ")
    assert "src/a.c.o(.text)" in first and "src/b.c.o" not in first.split("    .blob ")[0]
    assert "src/b.c.o(.data)" in second.split("}")[0] and "src/a.c.o" not in second


def test_a_setup_make_writes_local_mk_first(tmp_path, monkeypatch):
    """A clone has no local.mk: the tool paths (splat among them) reach make before any target runs."""
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path), host=SimpleNamespace(slots=0, workers=2))
    monkeypatch.setattr(build, "local_mk", lambda config: b"SPLAT := /venv/bin/splat\n")
    seen = []
    monkeypatch.setattr(build.process, "make", lambda c, t, j: seen.append((t, (tmp_path / "local.mk").read_bytes())))
    build.make(config, ["extract-us"])
    assert seen == [(["extract-us"], b"SPLAT := /venv/bin/splat\n")]


def test_owned_data_starts_its_own_piece_wherever_its_fact_lives(tmp_path):
    """A data name a file owns (in C or behind a stand-in) gets a piece: known from aliases.txt alone, from a row with
    no rom, or off a word boundary (the build fills each piece from the ROM). Off a word and owned by no file, a name
    stays folded."""
    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text(
        "d_1 = 0x80200500; // rom:0x1100\nd_norom = 0x80200600;\nd_odd = 0x80200703; // rom:0x1303\n"
        "d_fold = 0x80200802; // rom:0x1402\n"
    )
    facts = {
        "d_1": (0x80200500, 0x1100, None, "data"),
        "d_alias": (0x80200580, 0x1180, None, "data"),
        "d_norom": (0x80200600, 0x1200, None, "data"),
        "d_odd": (0x80200703, 0x1303, None, "data"),
        "d_fold": (0x80200802, 0x1402, None, "data"),
    }
    names = ("d_1", "d_alias", "d_norom", "d_odd")
    kinds = {"us": {n: ".data" for n in names} | {"f_code": ".text"}}
    facts["f_code"] = (0x80200900, 0x1500, None, "data")  # a code stand-in whose fact gives no type
    module = Module("src/d.c", "main", {"us": (*names, "f_code")}, (), {"us": (*names[2:], "f_code")}, kinds)
    held = Tree("digest", (module,), {"us": facts})
    out = build.splat_yaml(config, "us", held).decode()
    for row in ("[0x1100, data, main/d_1]", "[0x1180, data, main/d_alias]", "[0x1200, data, main/d_norom]"):
        assert row in out
    assert "[0x1303, data, main/d_odd]" in out and "d_fold" not in out and "f_code" not in out


def test_a_function_a_file_owns_starts_a_c_piece(tmp_path):
    """splat writes one file per `c` start: an owned function never reads as an entry inside the one before it."""
    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("f_1 = 0x80200400; // type:func rom:0x1000\n")
    facts = {"f_1": (0x80200400, 0x1000, None, "func"), "f_2": (0x80200410, 0x1010, None, "func")}
    module = Module("src/a.c", "main", {"us": ("f_1", "f_2")}, (), {"us": ()}, {"us": {"f_2": ".text"}})
    out = build.splat_yaml(config, "us", Tree("digest", (module,), {"us": facts})).decode()
    assert "      - [0x1000, c, main]\n      - [0x1010, c, main]\n" in out


def test_the_makefile_builds_and_verifies_standalone_without_our_tools(tmp_path):
    """make check is the ROMs, each checked by SHA-1; nothing calls unbake (the verdict is recorded from outside), and
    a version not yet extracted is extracted by the build itself."""
    text = build.makefile(make_config(tmp_path)).decode()
    assert "unbake" not in text.lower()
    assert "else ifeq ($(CHECK),rom)\ncheck: $(VERSIONS:%=build/%/$(NAME).z64)\n" in text
    assert "asm/%/.extracted: | roms/baserom.%.z64\n\t+$(Q)$(MAKE) --no-print-directory extract-$*\n" in text
    assert "build/%/undefined_syms_auto.txt build/%/undefined_funcs_auto.txt: | asm/%/.extracted\n" in text
    assert "asm/%.bin: | asm/$$(call ver,$$*)/.extracted\n" in text
    assert '|| { echo "$@: SHA-1 is not $(SHA1_$*)" >&2; exit 1; }' in text


def test_verify_records_real_rom_hashes_without_host_and_rejects_corruption(tmp_path, monkeypatch):
    from hashlib import sha1

    from unbake import report

    config = make_config(tmp_path)
    for version, expected in config.project.version.items():
        data = bytes(expected.size)
        expected.sha1 = sha1(data).hexdigest()
        path = tmp_path / "build" / version / "Game.z64"
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    monkeypatch.setattr(build.configuration, "load_project", lambda root: config.project)
    assert build.verify({"project": tmp_path}) == {"exact": ["eu-1", "us"]}
    held = make_tree()
    assert report.exact_versions(config, held) == {"eu-1", "us"}
    (tmp_path / "build/us/Game.z64").write_bytes(b"bad")
    assert report.exact_versions(config, held) == frozenset()
    with pytest.raises(Refusal, match=r"build\.rom"):
        build.verify({"project": tmp_path})
    assert report.exact_versions(config, held) == {"eu-1"}
    with pytest.raises(Refusal, match=r"build\.failed"):
        build.verify({"project": tmp_path, "status": 2})
    assert report.exact_versions(config, held) == frozenset()


def test_make_cache_hit_repairs_missing_report_verdict(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    held = make_tree()
    monkeypatch.setattr(build, "_make", lambda cfg, target: SimpleNamespace(exit=0, stderr=b""))
    build.make_check(config, held)
    (tmp_path / ".unbake/check.json").unlink()
    monkeypatch.setattr(build, "_make", Mock(side_effect=AssertionError("cached build must not rerun")))
    build.make_check(config, held)
    assert json.loads((tmp_path / ".unbake/check.json").read_bytes())["exact"] == ["eu-1", "us"]


def _said(monkeypatch) -> list[str]:
    said: list[str] = []
    monkeypatch.setattr(build.effort, "say", said.append)
    return said


def test_a_function_fact_with_no_rom_is_placed_by_its_segment(tmp_path, monkeypatch):
    """`f_1 = 0x80200410; // type:func` names no rom, yet the segment at 0x80200400 maps it to 0x1010: it starts a `c`
    piece there and a finding asks for the rom (a row with no rom was dropped from the yaml, so the function was never
    split from the data piece before it)."""
    said = _said(monkeypatch)
    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text(
        "d_0 = 0x80200400; // rom:0x1000\nf_1 = 0x80200410; // type:func\nabs_1 = 0x80000400; // type:func\n"
    )
    facts = {"d_0": (0x80200400, 0x1000, None, "data"), "f_1": (0x80200410, None, None, "func")}
    module = Module("src/a.c", "main", {"us": ("d_0",)}, (), {"us": ()}, {"us": {"d_0": ".data"}})
    out = build.splat_yaml(config, "us", Tree("digest", (module,), {"us": facts})).decode()
    assert "[0x1000, data, main/d_0]\n      - [0x1010, c, main]\n" in out
    assert said == ["us: f_1 at 0x80200410 names no rom; its segment maps it to 0x1010 (add rom:0x1010)"]


def test_a_data_piece_at_a_functions_rom_is_reported_as_shadowing_it(tmp_path, monkeypatch):
    """A four-byte data name at the first instruction of `f_1` wins the address, so splat wrote f_1's assembly from the
    second instruction (static_0_802A2270 in RageWars us-rev1); the generator now says so."""
    said = _said(monkeypatch)
    config = make_config(tmp_path)
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("f_1 = 0x80200410; // type:func rom:0x1010\n")
    facts = {"f_1": (0x80200410, 0x1010, None, "func"), "d_pre": (0x80200410, 0x1010, None, "data")}
    module = Module("src/a.c", "main", {"us": ("d_pre",)}, (), {"us": ("d_pre",)}, {"us": {"d_pre": ".data"}})
    build.splat_yaml(config, "us", Tree("digest", (module,), {"us": facts}))
    assert len(said) == 1 and said[0].startswith("us: f_1 at rom 0x1010 is shadowed by the data piece main/d_pre")


def test_the_canonical_name_of_a_function_the_c_spells_by_an_alias_has_that_owner():
    """The C defines `f_alias`; the fact names `f_canon` at the same address: both are one function and one owner,
    while `owned` stays strict (a second name never gets a piece of its own)."""
    module = Module("src/a.c", "main", {"us": ("f_alias",)}, (), {"us": ()}, {"us": {"f_alias": ".text"}})
    facts = {
        "f_alias": (0x80200400, 0x1000, None, "func"),
        "f_canon": (0x80200400, 0x1000, None, "func"),
        "f_other": (0x80200500, 0x1100, None, "func"),
    }
    held = Tree("digest", (module,), {"us": facts})
    assert held.owner("us", "f_canon") == held.owner("us", "f_alias") == "src/a.c"
    assert held.owned("us", "f_canon") is None and held.owner("us", "f_other") is None
    assert held.owner("us", "unknown") is None and held.owner("eu-1", "f_canon") is None


def test_a_literals_row_on_an_8_byte_boundary_asks_the_linker_for_that_alignment(tmp_path):
    """SUBALIGN(1) drops the section alignment the ROM kept as padding after the data before the file's literals."""
    tree = make_tree(
        both("src/a.c", "f_1", "literals_f_8", sections={"literals_f_8": ".rodata"}),
        both("src/b.c", "f_2", "literals_f_9", sections={"literals_f_9": ".rodata"}),
    )
    assert _body(build.linker(make_config(tmp_path), tree).decode()) == [
        "src/a.c.o(.text);",
        "src/b.c.o(.text);",
        ". = ALIGN(8) /* src/a.c.o */;",
        "src/a.c.o(.rodata .rdata);",
        "src/b.c.o(.rodata .rdata);",
    ]


@pytest.mark.usefixtures("no_apart")
def test_literals_after_other_files_data_apart_from_their_file_are_refused_where_the_toolchain_cannot_place_apart(
    tmp_path,
):
    tree = make_tree(
        both("src/a.c", "f_1", "d_2", "literals_f_5", sections={"d_2": ".rodata", "literals_f_5": ".rodata"}),
        both("src/b.c", "f_3", "d_4", sections={"d_4": ".rodata"}),
    )
    with pytest.raises(Refusal) as refused:
        build.linker(make_config(tmp_path), tree)
    assert refused.value.findings[0].key == "sync.split" and refused.value.findings[0].path == "src/a.c"


# ---------------------------------------------------------------- rule changes make cannot see
def _reflag_tree(tmp_path, makefile):
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us", "de"]))
    (tmp_path / "Makefile").write_text(makefile)
    for v in ("us", "de"):
        for unit in ("a", "b"):
            obj = tmp_path / f"build/{v}/src/{unit}.c.o"
            obj.parent.mkdir(parents=True, exist_ok=True)
            obj.write_bytes(b"o")
        (tmp_path / f"build/{v}/asm/d.s.o").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"build/{v}/asm/d.s.o").write_bytes(b"o")
    assert build.drop_reflagged(config) == 0  # the first build keeps its rules
    return config


ROW = "$(foreach v,$(VERSIONS),build/$(v)/src/a.c.o): CFLAGS := -O1\n"


def test_a_changed_per_file_row_drops_that_object_in_every_version(tmp_path):
    config = _reflag_tree(tmp_path, "CFLAGS := -O2\n" + ROW)
    assert build.drop_reflagged(config, ("# note\nCFLAGS := -O2\n" + ROW.replace("-O1", "-O3")).encode()) == 2
    assert not (tmp_path / "build/us/src/a.c.o").exists() and not (tmp_path / "build/de/src/a.c.o").exists()
    assert (tmp_path / "build/us/src/b.c.o").exists()
    assert build.drop_reflagged(config, ("# note\nCFLAGS := -O2\n" + ROW.replace("-O1", "-O3")).encode()) == 0


def test_a_changed_global_rule_drops_every_compiled_object(tmp_path):
    config = _reflag_tree(tmp_path, "CFLAGS := -O2\n" + ROW)
    (tmp_path / "Makefile").write_text("CFLAGS := -O3\n" + ROW)
    assert build.drop_reflagged(config) == 4
    assert (tmp_path / "build/us/asm/d.s.o").exists()  # an assembled file's rule names its one input


RULES = (
    "Q := @\nCFLAGS := -O2\nASFLAGS := -G0\n"
    "build/%.o: build/%.i.ok\n\t$(Q)cc $(CFLAGS) $< -o $@\n"
    "build/%.s.o: %.s\n\t$(Q)as $(ASFLAGS) $< -o $@\n"
)


def test_a_changed_assembler_recipe_keeps_every_compiled_object_and_remakes_the_assembled_ones(tmp_path):
    """A new variable only the .s.o recipe uses (a tool update adding assembler flags) reaches the assembled files,
    never the compiled objects: the C build stays warm."""
    config = _reflag_tree(tmp_path, RULES)
    changed = RULES.replace("ASFLAGS := -G0\n", "ASFLAGS := -G0\nASDEFS = --defsym A=1\n")
    changed = changed.replace("as $(ASFLAGS) $<", "as $(ASFLAGS) $(ASDEFS) $<")
    assert build.drop_reflagged(config, changed.encode()) == 0
    assert all((tmp_path / f"build/{v}/src/{u}.c.o").exists() for v in ("us", "de") for u in ("a", "b"))
    assert (tmp_path / "build/us/asm/d.s.o").stat().st_mtime_ns == build._LONG_AGO  # make assembles it again


def test_a_changed_variable_reaches_the_objects_of_the_rules_using_it(tmp_path):
    config = _reflag_tree(tmp_path, RULES)
    made = (tmp_path / "build/us/asm/d.s.o").stat().st_mtime_ns
    assert build.drop_reflagged(config, RULES.replace("-O2", "-O3").encode()) == 4
    assert (tmp_path / "build/us/asm/d.s.o").stat().st_mtime_ns == made


def test_dependency_lines_added_to_the_makefile_keep_every_compiled_object(tmp_path):
    """The deploy that adds the extracted-file prerequisites (the ASM_DEPS script and the lines that run and include
    it) must not read as a change nothing names: that would drop every compiled object of every tree."""
    config = _reflag_tree(tmp_path, RULES)
    block = (
        "define ASM_DEPS\nimport os, sys\nprint(os.getcwd(), sys.argv)\nendef\n"
        "$(shell mkdir -p build)\n$(file >build/asm-deps.py,$(ASM_DEPS))\n"
        "$(shell python3 -I build/asm-deps.py $(VERSIONS) >build/asm.deps)\n-include build/asm.deps\n"
    )
    assert build.drop_reflagged(config, (RULES + block).encode()) == 0
    assert all((tmp_path / f"build/{v}/src/{u}.c.o").exists() for v in ("us", "de") for u in ("a", "b"))
    assert (tmp_path / "build/us/asm/d.s.o").stat().st_mtime_ns != build._LONG_AGO


def test_a_changed_line_nothing_names_drops_every_compiled_object(tmp_path):
    config = _reflag_tree(tmp_path, RULES + "ifeq ($(X),1)\nY := 1\nendif\n")
    assert build.drop_reflagged(config, (RULES + "ifeq ($(X),2)\nY := 1\nendif\n").encode()) == 4


def test_rule_text_is_the_recipe_and_the_variables_it_uses():
    text = (
        "A := 1\nB := $(A) 2\nC := 3\nexport D\ndefine D\nx $(B)\nendef\nrun-%:\n\techo $(B) $$D\nother:\n\techo $(C)\n"
    )
    got = build.rule_text(text, "run-%")
    assert got is not None and "A := 1" in got and "B := $(A) 2" in got and "x $(B)" in got and "export D" in got
    assert "C := 3" not in got and "echo $(C)" not in got
    assert build.rule_text(text, "missing") is None


def _stamp(tmp_path, v, unit):
    return tmp_path / f"build/{v}/src/{unit}.c.i.ok"


def _stamped(tmp_path):
    for v in ("us", "de"):
        for unit in ("a", "b"):
            _stamp(tmp_path, v, unit).write_bytes(b"")


CPPROW = "$(foreach v,$(VERSIONS),build/$(v)/src/a.c.o): CPPFLAGS := -DA\n"


def test_a_changed_preprocessor_row_remakes_that_text_and_keeps_every_object(tmp_path):
    """A per-file CPPFLAGS row decides only the text: its stamps go (the .i is made again and compared), no object."""
    config = _reflag_tree(tmp_path, "CFLAGS := -O2\n" + CPPROW)
    _stamped(tmp_path)
    assert build.drop_reflagged(config, ("CFLAGS := -O2\n" + CPPROW.replace("-DA", "-DB")).encode()) == 0
    assert not _stamp(tmp_path, "us", "a").exists() and not _stamp(tmp_path, "de", "a").exists()
    assert _stamp(tmp_path, "us", "b").exists() and (tmp_path / "build/us/src/a.c.o").exists()


def test_a_changed_preprocess_recipe_remakes_every_text_and_keeps_every_object(tmp_path):
    recipe = "define PREPROCESS_x\n$(Q)cpp $(CPPFLAGS) $< > $(@:.ok=.new)\nendef\nCPPFLAGS := -P\n"
    config = _reflag_tree(tmp_path, recipe + ROW)
    _stamped(tmp_path)
    changed = recipe.replace("cpp ", "cpp -undef ").replace("-P", "-P -DX")
    assert build.drop_reflagged(config, (changed + ROW).encode()) == 0
    assert not any(_stamp(tmp_path, v, u).exists() for v in ("us", "de") for u in ("a", "b"))


def test_a_dropped_object_takes_its_stamp_with_it(tmp_path):
    config = _reflag_tree(tmp_path, "CFLAGS := -O2\n" + ROW)
    _stamped(tmp_path)
    assert build.drop_reflagged(config, ("CFLAGS := -O2\n" + ROW.replace("-O1", "-O3")).encode()) == 2
    assert not _stamp(tmp_path, "us", "a").exists() and _stamp(tmp_path, "us", "b").exists()


def test_drop_including_rebuilds_objects_whose_output_includes_a_name(tmp_path):
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us"]))
    for unit, text in (("a", '.include "nonmatchings/x/f.s"\n'), ("b", '.include "data/y.s"\n')):
        (tmp_path / f"build/us/{unit}.c.s").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"build/us/{unit}.c.s").write_text(text)
        (tmp_path / f"build/us/{unit}.c.o").write_bytes(b"o")
    (tmp_path / "build/us/c.c.o").write_bytes(b"o")  # no compiler output to read: rebuilt to be safe
    assert build.drop_including(config, "us", {"nonmatchings/x/f.s"}) == 2
    assert sorted(p.name for p in (tmp_path / "build/us").glob("*.o")) == ["b.c.o"]


def test_the_extraction_mark_holds_the_facts_hash_and_a_compile_against_other_facts_refuses(tmp_path):
    """REV1-ONE-43B: stale asm/ linked old names. The mark names the facts the extraction was made from; an object of a
    tree whose facts moved on without an extraction is refused with the cause named, not compiled."""
    import hashlib
    import subprocess
    import sys

    splat = _extract_world(tmp_path)
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/a.c").write_text("int a;\n")
    make = ["make", "--no-print-directory", f"SPLAT={sys.executable} {splat}", CCPY]
    done = subprocess.run([*make, "asm/us/.extracted"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    facts = b"".join((tmp_path / f"versions/us/{n}").read_bytes() for n in ("symbol_addrs.txt", "splat.yaml"))
    mark = tmp_path / "asm/us/.extracted"
    assert mark.read_text() == hashlib.sha1(facts).hexdigest() + "\n"
    # the facts move on while the mark stays: the object is refused, naming the version and both hashes
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("g = 0x1;\n")
    mark.write_text("0" * 40 + "\n")
    os.utime(tmp_path / "asm/us/.extracted", (time.time() + 100, time.time() + 100))  # no extraction is due
    (tmp_path / "build/.extract-src-us").write_text(
        hashlib.sha1(b"").hexdigest() + "\n"
    )  # a record that says the facts are the extracted ones (the other mechanism missed it)
    refused = subprocess.run(
        [*make, "-n", "-o", "asm/us/.extracted", "build/us/src/a.c.o", "STAMPS="],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    run = subprocess.run(
        [
            *make,
            "-o",
            "asm/us/.extracted",
            "build/us/src/a.c.o",
            "STAMPS=",
            "TC=gcc_2_8_1_sn64",
            _pp("PREPROCESS_gcc_2_8_1_sn64", "cat $<"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and "asm/us was extracted from other symbol_addrs.txt/splat.yaml" in run.stderr, (
        refused.stdout,
        run.stdout,
        run.stderr,
    )
    # an empty mark (an older extraction) is not checked
    mark.write_text("")
    run = subprocess.run(
        [
            *make,
            "-o",
            "asm/us/.extracted",
            "build/us/src/a.c.o",
            "STAMPS=",
            "TC=gcc_2_8_1_sn64",
            _pp("PREPROCESS_gcc_2_8_1_sn64", "cat $<"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert "was extracted from other" not in run.stderr


@pytest.mark.parametrize("tool", ["c", "python"])
def test_an_object_is_not_assembled_again_when_the_compilers_assembly_is_the_last_ones(tmp_path, tool):
    """The generated recipe of the real toolchain with stub tools: a preprocessed text that changed but compiles to the
    assembly already beside the object keeps the object (no assembler, no trim); other assembly assembles again."""
    import subprocess
    import sys

    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "src").mkdir()
    (tmp_path / "asm/us").mkdir(parents=True)
    (tmp_path / "asm/us/.extracted").touch()
    (tmp_path / "versions/us/text_sizes.mk").write_text("")
    cc = tmp_path / "cc.py"
    cc.write_text(
        "from pathlib import Path\nimport sys\n"
        "text = Path(sys.argv[-3]).read_text().splitlines(keepends=True)\n"
        "Path(sys.argv[-1]).write_text(''.join(t for t in text if not t.startswith('//')))\n"
        "with Path('compiled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
    )
    asm = tmp_path / "asm.py"
    asm.write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[-1]).write_bytes(Path(sys.argv[-3]).read_bytes())\n"
        "with Path('assembled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
    )
    trim = tmp_path / "trim.py"
    trim.write_text("from pathlib import Path\nwith Path('trimmed').open('a') as log: log.write('x\\n')\n")
    targets = ["build/us/src/a.c.o"]

    def make():
        for log in ("compiled", "assembled", "trimmed"):
            (tmp_path / log).write_text("")
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-j1",
                *targets,
                "STAMPS=",
                "TC=gcc_2_8_1_sn64",
                _pp("PREPROCESS_gcc_2_8_1_sn64", "cat $<"),
                f"CC_gcc_2_8_1_sn64={sys.executable} {cc}",
                f"ABUMAS={sys.executable} {asm}",
                "OBJCOPY=true",
                f"TRIM_TEXT={sys.executable} {trim}",
                CCPY,
                *(["HOSTCC=false"] if tool == "python" else []),  # no stamprec: the python object recipe
                "TEXTSIZE=0x10",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        compiled, assembled, trimmed = (
            len((tmp_path / log).read_text().splitlines()) for log in ("compiled", "assembled", "trimmed")
        )
        return [compiled, assembled, trimmed if tool == "python" else 0]  # stamprec trims in C: only the stub counts

    def edit(text):  # everything built ages first, so the edit is newer than every output
        now = time.time_ns()
        for path in tmp_path.rglob("*"):
            age = 10**10 if "build" in path.relative_to(tmp_path).parts else 2 * 10**10
            age += 10**9 if path.suffix == ".i" else 0
            os.utime(path, ns=(now - age, now - age))
        (tmp_path / "src/a.c").write_text(text)
        os.utime(tmp_path / "src/a.c", ns=(now - 5 * 10**9, now - 5 * 10**9))

    trims = 1 if tool == "python" else 0
    (tmp_path / "src/a.c").write_text("// note\nint a;\n")
    assert make() == [1, 1, trims]
    assert make() == [0, 0, 0]
    edit("// another note\nint a;\n")  # a new .i, the same assembly
    assert make() == [1, 0, 0]
    assert not (tmp_path / "build/us/src/a.c.o.same").exists() and not (tmp_path / "build/us/src/a.c.s.prev").exists()
    edit("// another note\nint a2;\n")
    assert make() == [1, 1, trims]
    assert make() == [0, 0, 0]


def test_a_project_without_a_non_matching_table_gets_no_nm_build(tmp_path):
    text = build.makefile(make_config(tmp_path)).decode()
    assert "nmraw" not in text and "NM_PICK" not in text and "DEFAULT_GOAL := nm" not in text
    assert "ver = $(firstword $(subst /, ,$1))" in text


def test_a_non_matching_table_adds_the_nm_build_with_the_files_rows(tmp_path):
    config = make_config(tmp_path, {"src/a.c": FileFacts(cflags=("-O1",))})
    config.project.non_matching = "DRAFT"
    text = build.makefile(config).decode()
    assert "-DDRAFT" in text and "ifeq ($(NON_MATCHING),1)\n.DEFAULT_GOAL := nm\nendif" in text
    assert "$(foreach v,$(VERSIONS),build/nmraw/$(v)/src/a.c.o): TC := gcc_2_8_1_sn64" in text
    assert "nm: $(VERSIONS:%=build/nm/%/$(NAME).z64)" in text
    assert "$(VERSIONS:%=build/nm/%/link.ld): build/nm/%/link.ld: build/%/link.ld" in text
    nm_link = text[text.index("build/nm/%/$(NAME).z64: build/nm/%/$(NAME).bin") :]
    assert "SHA-1" not in nm_link.split("\n\n")[0]  # an NM ROM matches no hash: nothing checks one


def elf(sections: dict[str, int], symbols: list[tuple[str, str, int]], fill: int = 0) -> bytes:
    """A minimal big-endian ELF32 object: PROGBITS sections of the given sizes and symbols (name, section, value)."""
    import struct

    names = list(sections)
    every = [*names, ".symtab", ".strtab", ".shstrtab"]
    shstr, strtab, index = b"\0", b"\0", {}
    for n in every:
        index[n] = len(shstr)
        shstr += n.encode() + b"\0"
    syms = b"\0" * 16
    for name, section, value in symbols:
        syms += struct.pack(">IIIBBH", len(strtab), value, 0, 0x12, 0, names.index(section) + 1)
        strtab += name.encode() + b"\0"
    blobs = [bytes([fill]) * sections[n] for n in names] + [syms, strtab, shstr]
    kinds = [1] * len(names) + [2, 3, 3]
    out, headers, offset = bytearray(52), [b"\0" * 40], 52
    for n, blob, kind in zip(every, blobs, kinds, strict=True):
        link = len(names) + 2 if kind == 2 else 0
        flags, entsize = (6 if kind == 1 else 0), (16 if kind == 2 else 0)
        headers.append(
            struct.pack(">10I", index[n], kind, flags, 0, offset, len(blob), link, int(kind == 2), 4, entsize)
        )
        out += blob
        offset += len(blob)
    out += b"".join(headers)
    ident = b"\x7fELF\x01\x02\x01" + bytes(9)
    count = len(headers)
    struct.pack_into(">16sHHIIIIIHHHHHH", out, 0, ident, 1, 8, 1, 0, 0, offset, 0, 52, 0, 0, 40, count, count - 1)
    return bytes(out)


@pytest.mark.parametrize(
    "raw,verdict",
    [
        (elf({".text": 16}, [("f", ".text", 0), ("g", ".text", 8)], fill=7), "draft"),
        (elf({".text": 16}, [("f", ".text", 0), ("g", ".text", 8), (".L12", ".text", 4)], fill=7), "draft"),
        (elf({".text": 16}, [("f", ".text", 0), ("g", ".text", 8)]), "same"),
        (elf({".text": 8}, [("f", ".text", 0)], fill=7), "draft-padded 8 bytes, 1 symbols move"),
        (elf({".text": 20}, [("f", ".text", 0), ("g", ".text", 8)], fill=7), "left-out .text:16->20"),
        (elf({".text": 16, ".rdata": 8}, [("f", ".text", 0), ("g", ".text", 8)], fill=7), "left-out .rdata:0->8"),
        (elf({".text": 16}, [("f", ".text", 0), ("g", ".text", 12)], fill=7), "draft-padded 0 bytes, 1 symbols move"),
    ],
)
def test_an_nm_object_replaces_the_matching_one_only_when_the_layout_stays(tmp_path, raw, verdict):
    """NM_PICK: the draft's object takes the place of the matching one when every section keeps its size and every
    symbol its offset; otherwise the matching object stays, and the reason is written beside the pick."""
    import shutil
    import subprocess
    import sys

    config = make_config(tmp_path)
    config.project.non_matching = "NON_MATCHING"
    text = build.makefile(config).decode()
    script = text[text.index("define NM_PICK\n") + len("define NM_PICK\n") : text.index("\nendef\nexport NM_PICK")]
    matching = elf({".text": 16}, [("f", ".text", 0), ("g", ".text", 8)])
    (tmp_path / "raw.o").write_bytes(raw)
    (tmp_path / "matching.o").write_bytes(matching)
    objcopy = shutil.which("mips-linux-gnu-objcopy") or shutil.which("objcopy")
    if verdict.startswith("draft-padded") and not (objcopy and shutil.which("mips-linux-gnu-ld")):
        pytest.skip("no objcopy and mips ld to pad with")
    run = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            "raw.o",
            "matching.o",
            "out.o",
            objcopy or "objcopy",
            shutil.which("mips-linux-gnu-ld") or "ld",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "out.o.pick").read_text() == f"{verdict} out.o\n"
    if verdict.startswith("draft-padded"):  # the draft's text, zero-filled to the matching unit's size
        subprocess.run([objcopy, "-O", "binary", "-j", ".text", "out.o", "text.bin"], cwd=tmp_path, check=True)
        filled = int(verdict.split()[1])
        assert (tmp_path / "text.bin").read_bytes() == bytes([7]) * (16 - filled) + bytes(filled)
    else:
        assert (tmp_path / "out.o").read_bytes() == (raw if verdict == "draft" else matching)


def test_the_nm_object_of_a_unit_builds_with_the_define_and_is_picked_against_the_matching_one(tmp_path):
    """make -n on the generated rules: the raw object compiles the same source with -DNON_MATCHING (the matching one
    without it), and the NM object is the pick between the two."""
    import subprocess

    config = make_config(tmp_path)
    config.project.non_matching = "NON_MATCHING"
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("int a;\n")
    (tmp_path / "asm/us").mkdir(parents=True)
    (tmp_path / "asm/us/.extracted").touch()
    (tmp_path / "versions/us/text_sizes.mk").write_text("build/us/src/a.c.o: TEXTSIZE := 0x10\n")
    run = subprocess.run(
        [
            *("make", "-n", "STAMPS=", "TC=stub", "PREPROCESS_stub=@echo pre $(DEFINES) $<"),
            *("COMPILE_stub=echo cc $@", "build/nm/us/src/a.c.o"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert any("pre -DVERSION_US -DNON_MATCHING src/a.c" in line for line in lines)
    assert any(re.search(r"pre -DVERSION_US\s+src/a.c", line) for line in lines)  # the matching object takes no define
    assert any(
        "build/nmraw/us/src/a.c.o build/us/src/a.c.o build/nm/us/src/a.c.o" in line and "$NM_PICK" in line
        for line in lines
    )
    assert "build/nmraw/us/src/a.c.o: TEXTSIZE := 0x10" in (tmp_path / "build/text_sizes_nmraw.mk").read_text()


def test_a_padded_nm_object_keeps_its_relocations(tmp_path):
    """Padding must not drop .rel.text: a draft whose calls lose their relocations links to jal 0."""
    import shutil
    import subprocess
    import sys

    tools = [shutil.which("mips-linux-gnu-" + t) for t in ("as", "ld", "objcopy", "readelf")]
    if not all(tools):
        pytest.skip("no mips binutils")
    as_, ld, objcopy, readelf = tools
    config = make_config(tmp_path)
    config.project.non_matching = "NON_MATCHING"
    text = build.makefile(config).decode()
    script = text[text.index("define NM_PICK\n") + len("define NM_PICK\n") : text.index("\nendef\nexport NM_PICK")]
    (tmp_path / "raw.s").write_text(".text\n.globl f\nf:\n jal g\n nop\n")
    (tmp_path / "match.s").write_text(".text\n.globl f\nf:\n jal g\n nop\n nop\n nop\n")
    for n in ("raw", "match"):
        subprocess.run([as_, "-march=vr4300", "-EB", "-o", f"{n}.o", f"{n}.s"], cwd=tmp_path, check=True)
    run = subprocess.run(
        [sys.executable, "-I", "-c", script, "raw.o", "match.o", "out.o", objcopy, ld],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert run.returncode == 0, run.stderr
    assert "draft-padded" in (tmp_path / "out.o.pick").read_text()
    rel = subprocess.run([readelf, "-r", "out.o"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "R_MIPS_26" in rel


def test_distinct_images_keep_physical_bounds_and_extraction_namespaces(tmp_path):
    config = make_config(tmp_path)
    config.project.version["us"].segments = (
        Segment("main", 0x1000, 0x80200400, "code"),
        Segment("old", 0x2000, 0x80200410, "code", "earlier"),
        Segment("old_tail", 0x2100, 0x80200510, "code", "earlier"),
        Segment("assets", 0x2200, None, "bin"),
    )
    held = make_tree()
    script = build.linker(config, held).decode()
    assert ".old 0x80200410 (INFO) : AT(0x2000)" in script
    assert ".main 0x80200400 : AT(0x1000)" in script
    assert 'ASSERT(SIZEOF(.main) <= 0x1000, "ROM extent: main")' in script
    assert 'ASSERT(SIZEOF(.old) <= 0x100, "ROM extent: old")' in script
    assert '"VMA overlap: old, old_tail"' in script
    assert '"VMA overlap: main, old"' not in script
    yaml = build.splat_yaml(config, "us", held).decode()
    assert yaml.count('exclusive_ram_id: "earlier"') == 2
    main = yaml.split("  - name: old\n")[0]
    assert 'exclusive_ram_id: "main"' in main
    make = build.makefile(config).decode()
    assert "--set-section-flags .old=alloc,load,readonly,code,contents --change-section-lma .old=0x2000" in make
    assert "$(OBJCOPY) $(ROM_SECTIONS_$*) -O binary" in make
    assert "--no-check-sections" not in make
    assert "--allow-multiple-definition" not in make


def test_distinct_image_emission_also_applies_to_nonmatching_rom(tmp_path):
    config = make_config(tmp_path)
    config.project.non_matching = "NON_MATCHING"
    config.project.version["us"].segments = (Segment("old", 0x1000, 0x80200400, "code", "old"),)
    text = build.makefile(config).decode()
    assert text.count("$(OBJCOPY) $(ROM_SECTIONS_$*) -O binary --gap-fill 0 $< $@") == 2


def test_separate_images_at_same_vma_do_not_share_namespace(tmp_path):
    config = make_config(tmp_path)
    config.project.version["us"].segments = (
        Segment("first", 0x1000, 0x80200400, "code", "first"),
        Segment("second", 0x2000, 0x80200400, "code", "second"),
    )
    script = build.linker(config, make_tree()).decode()
    assert "VMA overlap: first, second" not in script
    assert "ROM extent: first" in script and "ROM extent: second" in script
    yaml = build.splat_yaml(config, "us", make_tree()).decode()
    assert 'exclusive_ram_id: "first"' in yaml and 'exclusive_ram_id: "second"' in yaml


def _extract_world(tmp_path):
    """A tree whose fake splat appends a line to runs.log (and sleeps) each time it splits."""
    config = make_config(tmp_path)
    (tmp_path / "Makefile").write_bytes(build.makefile(config))
    (tmp_path / "roms").mkdir(exist_ok=True)
    (tmp_path / "roms/baserom.us.z64").write_bytes(bytes(64))
    (tmp_path / "versions/us/splat.yaml").write_text("")
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("f = 0x1;\n")
    splat = tmp_path / "splat.py"
    splat.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "if '--version' in sys.argv:\n"
        "    print('splat test')\n"
        "    sys.exit(0)\n"
        "time.sleep(0.5)\n"
        "with open('runs.log', 'a') as log:\n"
        "    log.write('split\\n')\n"
        "Path('asm/us/nonmatchings/seg').mkdir(parents=True, exist_ok=True)\n"
        "Path('asm/us/nonmatchings/seg/f.s').write_text(Path('versions/us/symbol_addrs.txt').read_text())\n"
    )
    return splat


def test_two_makes_extracting_one_version_split_it_once(tmp_path):
    """`make -j extract` and the build's own .extracted rule reach one recipe: the second waits, then sees the key."""
    import subprocess
    import sys

    splat = _extract_world(tmp_path)
    cmd = ["make", "--no-print-directory", "extract-us", f"SPLAT={sys.executable} {splat}"]
    pipe = subprocess.PIPE
    procs = [subprocess.Popen(cmd, cwd=tmp_path, stdout=pipe, stderr=pipe, text=True) for _ in range(3)]
    for proc in procs:
        _, err = proc.communicate()
        assert proc.returncode == 0, err
    assert (tmp_path / "runs.log").read_text() == "split\n"


def test_a_rebase_that_changed_the_facts_extracts_again_on_the_next_build(tmp_path):
    """asm/<v>/.extracted had only the ROM as prerequisite: symbol_addrs.txt changed and the build linked stale asm."""
    import subprocess
    import sys

    splat = _extract_world(tmp_path)
    cmd = ["make", "--no-print-directory", "asm/us/.extracted", f"SPLAT={sys.executable} {splat}"]

    def run():
        result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return (tmp_path / "runs.log").read_text().count("split")

    assert run() == 1
    stamp = tmp_path / "asm/us/.extracted"
    assert run() == 1  # nothing moved: no split, and no key check once the record is current
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("f = 0x1;\ng = 0x2;\n")
    assert run() == 2 and "g = 0x2" in (tmp_path / "asm/us/nonmatchings/seg/f.s").read_text()
    assert run() == 2 and stamp.is_file()
    # a tree recorded by an older Makefile (key but no source record) is checked once and keeps its extraction
    (tmp_path / "build/.extract-src-us").unlink()
    assert run() == 2 and (tmp_path / "build/.extract-src-us").is_file()


def _merged_unit() -> Tree:
    """src/a.c holds two functions with a pool each (so the ROM split them; the object is one), src/b.c one."""
    tree = make_tree(
        both(
            "src/a.c",
            "f_1",
            "literals_f_1",
            "f_2",
            "literals_f_2",
            sections={"literals_f_1": ".rodata", "literals_f_2": ".rodata"},
        ),
        both("src/b.c", "f_5"),
    )  # fmt: skip
    for v, pool in (("us", 0x8), ("us", 0x18), ("eu-1", 0x8), ("eu-1", 0x18)):  # each pool just before its function
        name = "literals_f_1" if pool == 0x8 else "literals_f_2"
        tree.symbols[v][name] = (0x80200000 + pool, 0x1000 + pool, 8, "data")
    return tree


def test_a_file_with_the_pools_of_several_functions_is_built_apart_and_the_others_are_not(tmp_path):
    rows = [
        (0x10, "src/a.c", ".rodata", "literals_f_1"),
        (0x20, "src/a.c", ".rodata", "literals_f_2"),
        (0x30, "src/b.c", ".rodata", "literals_f_5"),
    ]
    assert build._merged(rows) == {"src/a.c"}
    scattered = [
        (1, "x.c", ".text"),
        (2, "y.c", ".text"),
        (3, "x.c", ".text"),
        (4, "z.c", ".text"),
        (5, "z.c", ".rodata"),
    ]
    assert build._scattered(scattered) == {"x.c"}
    assert build._apart_paths(make_config(tmp_path), _merged_unit(), "us") == {"src/a.c"}


def test_text_sizes_mark_the_apart_object_to_be_built_with_a_section_per_function(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    assert "build/us/src/a.c.o: FSPLIT := fsplit" in build.text_sizes(config, _merged_unit(), "us").decode()
    real = build.configuration.toolchain
    monkeypatch.setattr(
        build.configuration, "toolchain", lambda id: {k: v for k, v in real(id).items() if k != "function_sections"}
    )
    assert (
        "FSPLIT" not in build.text_sizes(config, _merged_unit(), "us").decode()
    )  # a toolchain that cannot, is not asked


def test_an_apart_object_is_placed_function_by_function_at_the_offset_of_its_rom_address(tmp_path):
    text = build.linker(make_config(tmp_path), _merged_unit()).decode()
    body = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("src/", ". = ")) and ".bss" not in ln]
    assert body[:12] == [
        ". = MAX(., 0x8);",
        ". = DEFINED(__align8_literals_f_1) ? ALIGN(8) : .;",
        "src/a.c.o(.rodata.literals_f_1 .rdata.literals_f_1);",
        ". = MAX(., 0x10);",
        "src/a.c.o(.text.f_1);",
        ". = MAX(., 0x18);",
        ". = DEFINED(__align8_literals_f_2) ? ALIGN(8) : .;",
        "src/a.c.o(.rodata.literals_f_2 .rdata.literals_f_2);",
        ". = MAX(., 0x20);",
        "src/a.c.o(.text.f_2);",
        ". = MAX(., 0x50);",
        "src/b.c.o(.text);",
    ]
    assert "src/b.c.o(.text);" in body  # an object whole in the ROM keeps its one `.text`


def test_every_object_is_placed_at_the_offset_of_its_first_row_so_a_dropped_gap_stays_zeros(tmp_path):
    tree = make_tree(
        both("src/a.c", "f_1", "f_2"), both("src/b.c", "f_4")
    )  # rows are 0x10 long: 0x30 is owned by no file
    text = build.linker(make_config(tmp_path), tree).decode()
    body = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("src/", ". = ")) and ".bss" not in ln]
    assert body[:4] == [". = MAX(., 0x10);", "src/a.c.o(.text);", ". = MAX(., 0x40);", "src/b.c.o(.text);"]


def test_a_function_with_no_pool_row_of_its_own_has_its_pool_kept_with_the_function_before(tmp_path):
    functions, rows = ["f_1", "f_2", "f_3"], {"literals_f_1", "literals_f_3"}
    assert build._covered(functions, rows, "literals_f_1") == ".rodata.literals_f_1 .rodata.literals_f_2"
    assert build._covered(functions, rows, "literals_f_3") == ".rodata.literals_f_3"
    rows = {"literals_f_1"}
    assert (
        build._covered(functions, rows, "literals_f_1")
        == ".rodata.literals_f_1 .rodata.literals_f_2 .rodata.literals_f_3"
    )
    assert build._covered(functions, rows, "d_9") == ".rodata.d_9"  # not a pool: its own section


def _fsplit(tmp_path, text: str) -> list[str]:
    import subprocess
    import sys

    (tmp_path / "in.s").write_text(text)
    script = adapters.fsplit_script()
    subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path / "in.s"), str(tmp_path / "out.s")], check=True)
    return (tmp_path / "out.s").read_text().splitlines()


def test_fsplit_pass_gives_each_function_and_each_read_only_block_a_section_of_its_own(tmp_path):
    out = _fsplit(
        tmp_path,
        '.section .f_1,"ax"\nf_1:\n lui $4, %hi(.LC0)\n.section .rdata\n.align 3\nD_blob_1:\n.word 1\n'
        '.section .rdata\n.align 2\n.LC0:\n.word 2\n.section .f_2,"ax"\nf_2:\n jr $31\n',
    )  # fmt: skip
    sections = [line.split("\t", 1)[1] for line in out if line.startswith(".section")]
    assert sections == [
        '.text.f_1,"ax",@progbits',
        '.rdata.D_blob_1,"a",@progbits',
        '.rdata.literals_f_1,"a",@progbits',
        '.text.f_2,"ax",@progbits',
    ]
    assert "\t.set __align8_D_blob_1, 0" in out  # a block aligned to 8 says so (SUBALIGN(1) drops the section's)
    assert not any("__align8_literals_f_1" in line for line in out)
