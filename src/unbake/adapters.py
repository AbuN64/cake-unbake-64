"""Template-driven toolchain, assembler and linker adapters; all native behaviour comes from toolchains.toml."""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

import abucache
import abucache.compile
from abucache import link as linking
from abucache.compile import Step

from unbake import config as configuration
from unbake import effort, process
from unbake.contracts import Assembler, Config, Finding, Json, Linker, NativeResult, Placement, Recipe, Refusal, macro

_VARIABLES = ('"$1"', '"$3"', '"$3.i"', '"$3.s"')


class _Joined(dict[str, Any]):
    def __getitem__(self, key: str) -> str:
        return " ".join(super().__getitem__(key))


def render(template: Sequence[str], values: Mapping[str, Sequence[str]]) -> list[str]:
    """A word exactly `{x}` splices the list values[x]; other words are formatted with lists joined by spaces."""
    joined = _Joined(values)
    words: list[str] = []
    for word in template:
        if len(word) > 2 and word[0] == "{" and word[-1] == "}" and word[1:-1].isidentifier():
            words.extend(values[word[1:-1]])
        else:
            words.append(word.format_map(joined))
    return words


def assemble_template(row: Json | None) -> tuple[tuple[str, ...], str]:
    if row is not None and row["emits_asm"] and "assemble" in row:
        return tuple(row["assemble"]), "as"
    return ("{as}", "{asflags}", "{includes}", "{source}", "-o", "{out}"), "mips_as"


def host_tools(config: Config) -> dict[str, str]:
    return {
        k: str(process.tool(config, k))
        for k in ("cpp", "mips_as", "mips_ld", "mips_objcopy", "abumas", "armips")
        if k in config.host.tools
    }


def _tail(result: NativeResult) -> str:
    lines = result.stderr.decode(errors="replace").splitlines()
    noisy = any("warning" in ln for ln in lines)  # a compile with warnings keeps its error lines, however early
    errors = [ln for ln in lines if ln.strip() and "warning" not in ln and not ln.rstrip().endswith(":")]
    return "\n".join(errors if noisy and errors else lines[-20:]) or f"exit {result.exit}"


def _assembled(result: NativeResult, source: Path) -> NativeResult:
    if result.exit != 0 or result.signal is not None:
        raise Refusal(Finding("assemble.error", reason=_tail(result), path=str(source)))
    return result


def _unknown(reason: str) -> Refusal:
    return Refusal(Finding("adapter.unknown", reason=reason))


def row(id: str) -> Json:
    rows = configuration.toolchains()
    if id not in rows:
        raise _unknown(f"toolchain {id} is not in toolchains.toml")
    chosen: Json = rows[id]
    return chosen


class GnuAssembler:
    def __init__(self, config: Config) -> None:
        self.config = config

    def assemble(self, source: Path, recipe: Recipe, include: Sequence[Path], out: Path) -> NativeResult:
        with effort.stage("adapters.assemble"):
            argv = render(
                assemble_template(None)[0],
                {
                    "as": [str(process.tool(self.config, "mips_as"))],
                    "asflags": recipe.asflags,
                    "includes": [f"-I{p}" for p in include],
                    "source": [str(source)],
                    "out": [str(out)],
                },
            )
            result = process.run(
                "assemble",
                argv,
                self.config.project.root,
                outputs={"object": out},
                tmp=process.scratch(self.config.project.root),
            )
            return _assembled(result, source)


class Armips:
    def __init__(self, config: Config) -> None:
        self.config = config

    def assemble(self, source: Path, recipe: Recipe, include: Sequence[Path], out: Path) -> NativeResult:
        with effort.stage("adapters.assemble"):
            root = self.config.project.root
            # Source emits to the wrapper-selected output; never trusts a stale declared file.
            wrapper = out.with_suffix(".armips.s")
            text = source.read_text()
            text = re.sub(r"(?m)^\s*\.create\s+[^\n]+", "", text)
            text = re.sub(r"(?m)^\s*\.close\s*$", "", text)
            wrapper.write_text(f'.create "{out}", 0\n' + text + "\n.close\n")
            argv = [str(process.tool(self.config, "armips", kind="resource")), str(wrapper), "-root", str(root)]
            result = process.run("armips", argv, root, outputs={"object": out}, tmp=process.scratch(root))
            return _assembled(result, source)


class TemplateToolchain:
    def __init__(self, config: Config, id: str, row: Json) -> None:
        self.config, self.id, self.row = config, id, row
        base = config.host.toolchain_root / id
        self._tools = {
            "cc": [str(base / row["cc"])],
            "as": [str(base / row["as"])]
            if "as" in row
            else [str(config.host.tools[row["as_host"]])]
            if row["as_host"] in config.host.tools
            else [],
            **{k: [str(process.tool(config, k))] for k in ("cpp", "abumas") if k in config.host.tools},
        }

    def tool_files(self) -> list[Path]:
        """The files of the tools that the compile and assemble templates name."""
        template = self.row["assemble"] if "assemble" in self.row else assemble_template(None)[0]
        files: list[Path] = []
        words = {w[1:-1] for t in (self.row["compile"], template) for w in t if re.fullmatch(r"\{\w+\}", w)}
        for name in sorted(words):
            if self._tools.get(name):
                files.append(Path(self._tools[name][0]))
            elif name == "as":
                files.append(process.tool(self.config, self.row.get("as_host", "mips_as")))
            elif name in self.config.host.tools:
                files.append(process.tool(self.config, name))
        if self.row.get("rdata_alloc"):
            files.append(process.tool(self.config, "mips_objcopy"))
        return files

    def values(self, **given: Sequence[str]) -> dict[str, Sequence[str]]:
        return {**self._tools, **given}

    def argv(self, phase: str, **given: Sequence[str]) -> list[str]:
        values = self.values(**given)
        for word in self.row[phase]:
            for key in re.findall(r"{(\w+)}", word):
                if key not in values or (key == "as" and not values[key]):
                    values[key] = [str(process.tool(self.config, self.row.get("as_host", key) if key == "as" else key))]
        return render(self.row[phase], values)

    def preprocess(
        self, source: Path, recipe: Recipe, version: str, include: Sequence[Path], out: Path
    ) -> NativeResult:
        with effort.stage("adapters.preprocess"):
            project = self.config.project
            if version not in project.version:
                raise _unknown(f"version {version}")
            argv = self.argv(
                "preprocess",
                cppflags=recipe.cppflags,
                defines=[f"-D{macro(version)}"],
                includes=[f"-I{p}" for p in include],
                source=[str(source)],
            )
            result = process.run(
                "preprocess",
                argv,
                project.root,
                stdout_path=out,
                outputs={"preprocessed": out},
                tmp=process.scratch(project.root),
            )
            self._raise(self.diagnose(result))
            return result

    def compile(self, preprocessed: Path, recipe: Recipe, out: Path) -> NativeResult:
        with effort.stage("adapters.compile"):
            argv = self.argv(
                "compile",
                codegen=recipe.cflags,
                name=[out.with_suffix("").name],
                source=[str(preprocessed)],
                out=[str(out)],
            )
            role = "asm" if self.row["emits_asm"] else "object"
            result = process.run(
                "compile", argv, out.parent, outputs={role: out}, tmp=process.scratch(self.config.project.root)
            )
            self._raise(self.diagnose(result))
            return result

    def assemble(self, source: Path, recipe: Recipe, include: Sequence[Path], out: Path) -> NativeResult:
        with effort.stage("adapters.assemble"):
            if "assemble" not in self.row:
                return GnuAssembler(self.config).assemble(source, recipe, include, out)
            includes = [f"-I{p}" for p in include]
            argv = self.argv(
                "assemble",
                asflags=_asflags(self.row, recipe, includes),
                includes=includes,
                name=[out.with_suffix("").name],
                source=[str(source)],
                out=[str(out)],
            )
            result = process.run(
                "assemble", argv, out.parent, outputs={"object": out}, tmp=process.scratch(self.config.project.root)
            )
            return _assembled(result, source)

    def check(self, step: Step, result: NativeResult) -> None:
        """Refuse a failed compile or assemble step of an abucache chain, named as the direct calls name it."""
        if step.phase == "compile":
            self._raise(self.diagnose(result))
        else:
            _assembled(result, Path(step.out))

    def diagnose(self, result: NativeResult) -> tuple[Finding, ...]:
        role = "preprocess" if "preprocessed" in result.outputs else "compile"
        stderr = result.stderr.decode(errors="replace")
        if result.exit != 0:
            return (Finding(f"{role}.error", reason=_tail(result)),)
        pattern = self.row.get("error_pattern")
        match = re.search(pattern, stderr, re.MULTILINE) if pattern else None
        if match:
            start = stderr.rfind("\n", 0, match.start()) + 1
            end = stderr.find("\n", match.end())
            return (Finding("preprocess.error", reason=stderr[start : end if end >= 0 else None].strip()),)
        return ()

    def refused(self, stderr: str) -> list[str]:
        """The compiler lines that spell a refused view (data: refused_warnings), never a failure by themselves."""
        return [
            line.strip()
            for line in stderr.splitlines()
            if any(re.search(warning, line) for warning in self.row["refused_warnings"])
        ]

    @staticmethod
    def _raise(findings: tuple[Finding, ...]) -> None:
        if findings:
            raise Refusal(*findings)


def link_script(sections: Sequence[Placement], symbols: Path, trim: bool = False) -> str:
    return linking.link_script(sections, str(symbols), configuration.sections(), trim=trim)


def link_commands(
    tools: Mapping[str, str], obj: Path, sections: Sequence[Placement], work: Path, trim: bool, *, score: bool = False
) -> tuple[tuple[str, ...], ...]:
    return linking.link_commands(tools, obj, sections, work, trim, score=score)


class N64Link:
    def __init__(self, config: Config) -> None:
        self.config = config

    def link(self, obj: Path, sections: Sequence[Placement], symbols: Path, work: Path, trim: bool) -> NativeResult:
        with effort.stage("adapters.link"):
            root = self.config.project.root
            tools = {k: str(process.tool(self.config, k)) for k in ("mips_ld", "mips_objcopy")}

            def run(phase: str, argv: Sequence[str], cwd: Path) -> NativeResult:
                return process.run("link", argv, root, tmp=process.scratch(root))

            try:
                done = linking.link(
                    run, tools, obj, sections, symbols, work, trim, configuration.sections(), score=True
                )
            except linking.LinkFailed as error:
                reason = _tail(error.result) if error.result is not None else error.reason
                raise Refusal(Finding("link.error", reason, versions=(sections[0].version,))) from error
            result: NativeResult = done.result
            return replace(result, outputs=done.outputs)


def _asflags(row: Json, recipe: Recipe, includes: Sequence[str]) -> list[str]:
    """The assembler flags of a row's assemble template: a template without {includes} takes the include directories
    after its flags, as the generated Makefile passes $(ASFLAGS) $(ASINC), so a stand-in's .include resolves."""
    return [*recipe.asflags, *(() if "{includes}" in row["assemble"] else includes)]


def hilo_hazard_script() -> str:
    """The python of the generated Makefile's HILO_HAZARD: the one text both builds run."""
    text = configuration.template("Makefile.in")
    return text.split("define HILO_HAZARD\n", 1)[1].split("\nendef", 1)[0] + "\n"


def fsplit_script() -> str:
    """The python of the generated Makefile's FSPLIT_PASS: the one text the build runs on an object whose functions the
    ROM places apart."""
    text = configuration.template("Makefile.in")
    return text.split("define FSPLIT_PASS\n", 1)[1].split("\nendef", 1)[0] + "\n"


def strip_align_script() -> str:
    """The python of the generated Makefile's STRIP_ALIGN: the one text both builds run, so what a measure assembles is
    what the build assembles."""
    text = configuration.template("Makefile.in")
    return text.split("define STRIP_ALIGN\n", 1)[1].split("\nendef", 1)[0] + "\n"


def chain_steps(
    config: Config, id: str, recipe: Recipe, include: Sequence[Path], stem: str, align: int = 0, hilo: bool = False
) -> tuple[Step, ...]:
    """The compile and assemble commands of a unit with {in} and {out} for their files, for abucache's compile cache.
    `align` is the address mod 8 the ROM puts the unit's data at (the Makefile's NOALIGN): the assembly drops the
    alignment that start does not meet before it is assembled. `hilo` is the unit's file row hilo_nops: the Makefile's
    HILO_HAZARD pass puts two nops between an mfhi/mflo and an adjacent mult/div before the assembly is assembled."""
    chain = toolchain(config, id)
    row, includes = chain.row, [f"-I{p}" for p in include]
    steps = [
        Step(
            "compile",
            "source.s" if row["emits_asm"] else "source.o",
            tuple(chain.argv("compile", codegen=recipe.cflags, name=[stem], source=["{in}"], out=["{out}"])),
        )
    ]
    if row["emits_asm"]:
        if align:
            steps.append(
                Step("align", "source.s", (sys.executable, "-I", "-c", strip_align_script(), "{in}", str(align)))
            )
        if hilo:
            steps.append(Step("hilo", "source.s", (sys.executable, "-I", "-c", hilo_hazard_script(), "{in}")))
        argv = (
            chain.argv(
                "assemble",
                asflags=_asflags(row, recipe, includes),
                includes=includes,
                name=[stem],
                source=["{in}"],
                out=["{out}"],
            )
            if "assemble" in row
            else render(
                assemble_template(None)[0],
                {
                    "as": [str(process.tool(config, "mips_as"))],
                    "asflags": recipe.asflags,
                    "includes": includes,
                    "source": ["{in}"],
                    "out": ["{out}"],
                },
            )
        )
        steps.append(Step("assemble", "source.o", tuple(argv)))
        if row.get(
            "rdata_alloc"
        ):  # the Makefile's own step: the assembler leaves .rdata unallocated and the link drops it
            flags = (".rdata=alloc,load,readonly,data,contents",)
            objcopy = str(process.tool(config, "mips_objcopy"))
            steps.append(Step("objcopy", "allocated.o", (objcopy, "--set-section-flags", *flags, "{in}", "{out}")))
    return tuple(steps)


def tool_identity(config: Config, id: str) -> str:
    """The hash of the bytes of every tool the compile and assemble commands name, and of the row's pins."""
    chain = toolchain(config, id)
    return abucache.compile.identity(chain.tool_files(), [f"{k}={v}" for k, v in sorted(chain.row["pins"].items())])


def _pin() -> str:
    """The abucache version this unbake's installed requirement names (its wheel URL carries it)."""
    requirement = next(r for r in metadata.requires("unbake") or () if r.startswith("abucache"))
    found = re.search(r"abucache-([0-9][^-]*)-", requirement)
    return found[1] if found else ""


def check_abucache() -> None:
    """The installed abucache is the one version this unbake's own requirement names."""
    pin = _pin()
    if abucache.__version__ != pin:
        raise Refusal(
            Finding(
                "tool.version",
                path="pyproject.toml:dependencies",
                reason=f"abucache version mismatch: expected {pin}, found {abucache.__version__}",
            )
        )


def toolchain(config: Config, id: str) -> TemplateToolchain:
    return TemplateToolchain(config, id, row(id))


def assembler(config: Config, kind: str) -> Assembler:
    kinds: dict[str, Callable[[Config], Assembler]] = {"gnu": GnuAssembler, "armips": Armips}
    if kind not in kinds:
        raise _unknown(f"assembler {kind}")
    return kinds[kind](config)


def linker(config: Config) -> Linker:
    return N64Link(config)


def _asm_includes(config: Config, version: str) -> list[str]:
    """Where the build's assembler finds the `.include` lines a C unit's INCLUDE_ASM leaves in its assembly: the
    version's extraction, its include folder and the project's, absolute because a script runs from any directory."""
    root = config.project.root
    return [f"-I{root / 'asm' / version}", f"-I{root / 'asm' / version / 'include'}", f"-I{root / 'include'}"]


def script(config: Config, recipe: Recipe, version: str) -> str:
    """POSIX sh compiling "$1" to "$3" as a compile script; every word quoted except the "$n" words."""
    with effort.stage("adapters.script"):
        chain = toolchain(config, recipe.toolchain)
        if version not in config.project.version:
            raise _unknown(f"version {version}")
        asm = chain.row["emits_asm"]
        steps = [
            (
                chain.argv(
                    "preprocess",
                    cppflags=recipe.cppflags,
                    defines=[f"-D{macro(version)}"],
                    includes=[],
                    source=['"$1"'],
                ),
                ' > "$3.i"',
            ),
            (
                chain.argv(
                    "compile",
                    codegen=recipe.cflags,
                    name=["function"],
                    source=['"$3.i"'],
                    out=['"$3.s"' if asm else '"$3"'],
                ),
                "",
            ),
        ]
        if asm:
            if "assemble" not in chain.row:
                raise _unknown(f"toolchain {recipe.toolchain} has no assemble template")
            steps.append(
                (
                    chain.argv(
                        "assemble",
                        asflags=[*recipe.asflags, *_asm_includes(config, version)],  # the build's ASFLAGS + ASINC
                        includes=[],
                        name=["function"],
                        source=['"$3.s"'],
                        out=['"$3"'],
                    ),
                    "",
                )
            )
        lines = ["#!/bin/sh", "set -e", """trap 'rm -f "$3.i" "$3.s"' EXIT"""]
        for argv, redirect in steps:
            lines.append(" ".join(w if w in _VARIABLES else shlex.quote(w) for w in argv) + redirect)
        return "\n".join(lines) + "\n"
