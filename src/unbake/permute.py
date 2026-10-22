"""decomp-permuter on one function: a measured context in, the best permuted bodies out as a candidate file."""

from __future__ import annotations

import os
import re
import signal
from pathlib import Path
from typing import Any

from unbake import process, store
from unbake.contracts import Config, Finding, Json, Refusal


# passes whose output no original source has (masks, ^0, empty statements, dummy blocks): never proposed
_JUNK = (
    "add_mask xor_zero refer_to_var empty_stmt mult_zero dummy_comma_expr add_self_assignment pad_var_decl "
    "alias_array ins_block remove_ast duplicate_assignment var_cond_block"
).split()


def span(text: str, name: str) -> tuple[int, int] | None:
    """The [start, end) of `name`'s definition (its head line to the closing brace) in C text, None when it has none."""
    head = re.compile(rf"^[A-Za-z_][^\n;{{}}=]*\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", re.M)
    for found in head.finditer(text):
        depth, at = 0, found.end() - 1
        while at < len(text):
            depth += (text[at] == "{") - (text[at] == "}")
            at += 1
            if depth == 0:
                return found.start(), at
    return None


def replace(source: str, name: str, body: str) -> str:
    """`source` with `name`'s definition replaced by `body`; a refusal when `source` defines no such function."""
    where = span(source, name)
    if where is None:
        raise Refusal(Finding("land.request", f"{name} has no definition to replace", unit=name))
    return source[: where[0]] + body.strip() + source[where[1] :]


def _script(root: Path, script: str) -> str:
    """The build's compile script run from the project root with "$1" and "$3" made absolute first."""
    head = f'set -- "$(readlink -f "$1")" "$2" "$(readlink -f "$3")"; cd {root}\n'
    return "#!/bin/sh\n" + head + script.removeprefix("#!/bin/sh\n") + "\n"


def search(config: Config, name: str, ctx: Json, params: Json, work: Path) -> Json:
    """Runs the permuter for `params["permute"]` seconds on the function's context (`api.build_context`) and returns
    {base_score, best_score, function, diff, outputs}: scores are the permuter's (lower is closer to the target), the
    function is the best output's body (macros expanded, as the compiler read it) and diff its change from the base."""
    permuter = params.get("permuter") or os.environ.get("UNBAKE_PERMUTER", "")
    if not permuter or not Path(permuter).is_file():
        raise Refusal(
            Finding("native.missing_tool", "no permuter: pass --permuter PATH or set UNBAKE_PERMUTER", path=permuter)
        )
    root, folder = config.project.root, work / "permute"
    folder.mkdir(parents=True, exist_ok=True)
    store.write(folder / "base.c", ctx["source"].encode())
    store.write(folder / "compile.sh", _script(root, ctx["script"]).encode())
    (folder / "compile.sh").chmod(0o755)
    weights = "".join(f"perm_{p} = 0\n" for p in _JUNK)
    settings = f'func_name = "{name}"\ncompiler_type = "gcc"\n[weight_overrides]\n{weights}'
    store.write(folder / "settings.toml", settings.encode())
    assemble = [a.replace("{out}", str(folder / "target.o")) for a in ctx["assemble"]]
    done = process.run("permute.target", assemble, root, tmp=work / "tmp")
    if done.exit != 0:
        raise Refusal(Finding("native.exit", f"assembling the target failed: {done.stderr.decode()[-300:]}", unit=name))
    jobs = str(params.get("jobs") or min(8, config.host.workers))
    argv = ["/usr/bin/nice", "-n", "10", permuter, "-j", jobs, "--best-only", "--quiet", str(folder)]
    ran = process.run("permute.run", argv, folder, tmp=work / "tmp", timeout=float(params["permute"]))
    said = re.search(r"base score = (\d+)", ran.stdout.decode(errors="replace"))
    if said is None and ran.signal != signal.SIGKILL:
        raise Refusal(
            Finding("native.exit", f"permuter said: {ran.stderr.decode()[-300:] or ran.stdout.decode()[-300:]}")
        )
    base = int(said[1]) if said else None
    rows = sorted(
        (int((d / "score.txt").read_text()), d)
        for d in folder.glob("output-*")
        if (d / "score.txt").is_file() and (d / "source.c").is_file()
    )
    result: dict[str, Any] = {"base_score": base, "best_score": None, "function": "", "diff": "", "outputs": len(rows)}
    if rows and (base is None or rows[0][0] < base):
        score, best = rows[0]
        where = span(source := (best / "source.c").read_text(), name)
        result |= {
            "best_score": score,
            "function": source[where[0] : where[1]] if where else "",
            "diff": (best / "diff.txt").read_text()[:4000] if (best / "diff.txt").is_file() else "",
        }
    return result


def candidate(source: str, name: str, found: Json) -> str | None:
    """The candidate text with the permuted function in place of its own, or None when nothing beat the base."""
    return replace(source, name, found["function"]) if found["function"] else None
