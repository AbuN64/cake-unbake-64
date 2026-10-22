"""decomp-permuter on one function: a measured context in, the best permuted bodies out as a candidate file."""

from __future__ import annotations

import os
import re
import shlex
import signal
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unbake import effort, process, store
from unbake.contracts import Config, Finding, Json, Refusal

# passes whose output no original source has (masks, ^0, empty statements, dummy blocks): never proposed
_JUNK = (
    "add_mask", "xor_zero", "refer_to_var", "empty_stmt", "mult_zero", "dummy_comma_expr", "add_self_assignment",
    "pad_var_decl", "alias_array", "ins_block", "remove_ast", "duplicate_assignment", "var_cond_block",
)  # fmt: skip


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


_ATTRIBUTE = re.compile(r"\}\s*(__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\))")
_BRACKET = re.compile(r"\[[^\]\n]*\]")
_SIZEOF = re.compile(r"\bsizeof\s*\(\s*(\w+)\s*\)")


def _shield(source: str) -> tuple[str, list[str]]:
    """The source the permuter's C parser can round-trip, and the -D options that put the original text back when the
    result is preprocessed again. Its parser reads `} __attribute__((aligned(4)));` as a lost attribute and
    `[N - sizeof(A) - sizeof(B)]` as a cast; both become one identifier that the compile script defines as the
    original, so every permuted body is compiled against the layout the build has."""
    defines: dict[str, str] = {}

    def attribute(found: re.Match[str]) -> str:
        name = f"UNBAKE_ATTR_{len(defines)}"
        defines[name] = found[1]
        return f"}} {name}"

    def sizes(found: re.Match[str]) -> str:
        def one(word: re.Match[str]) -> str:
            name = f"UNBAKE_SIZEOF_{word[1]}"
            defines[name] = f"sizeof({word[1]})"
            return name

        return _SIZEOF.sub(one, found[0]) if "-" in found[0] else found[0]

    text = _BRACKET.sub(sizes, _ATTRIBUTE.sub(attribute, source))
    return text, [f"-D{name}={body}" for name, body in defines.items()]


def _script(root: Path, script: str, defines: Sequence[str] = ()) -> str:
    """The build's compile script run from the project root with "$1" and "$3" made absolute first; `defines` go to
    its preprocessor."""
    if defines:
        script = script.replace('"$1"', " ".join(shlex.quote(d) for d in defines) + ' "$1"', 1)
    head = f'set -- "$(readlink -f "$1")" "$2" "$(readlink -f "$3")"; cd {root}\n'
    return "#!/bin/sh\n" + head + script.removeprefix("#!/bin/sh\n") + "\n"


def _python(permuter: str) -> list[str]:
    """The interpreter that runs the permuter script: UNBAKE_PERMUTER_PYTHON, else the python of the venv beside the
    user's home named venv-permuter or beside the script, else none (the script's own shebang). The shebang's python
    is the system one, which holds none of the permuter's requirements."""
    given = os.environ.get("UNBAKE_PERMUTER_PYTHON", "")
    home, here = Path.home(), Path(permuter).resolve().parent
    for python in (given, home / "venv-permuter/bin/python", here / "venv/bin/python", here / ".venv/bin/python"):
        if python and Path(python).is_file() and os.access(python, os.X_OK):
            return [str(python)]
    return []


CAP = 2  # permuter runs at once on one host (UNBAKE_PERMUTER_CAP): each runs -j workers, more starve every other lane
QUEUE_S = 600.0  # seconds a run queues for a place before it refuses, naming who holds them


def _is_run(words: list[str]) -> bool:
    """Whether a command line is the permuter itself: the script, or a python running it."""
    names = [Path(w).name for w in words[:2]]
    return bool(names) and (
        names[0] == "permuter.py" or (names[0].startswith("python") and names[1:] == ["permuter.py"])
    )


def _reservations() -> Path:
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/unbake-{os.getuid()}")
    return base / "unbake-permuter"


def _running(proc: Path = Path("/proc"), reserved: Path | None = None) -> list[tuple[int, str]]:
    """The permuter runs on this host now, as (pid, command): processes whose command line names permuter.py and
    whose parent's does not (a run's workers are its children), and the unbake commands that reserved a place and
    have not started theirs yet."""
    roots: dict[int, tuple[int, str]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if _is_run(command.split()):
            roots[int(entry.name)] = (parent, command)
    out = [(pid, cmd[:120]) for pid, (parent, cmd) in sorted(roots.items()) if parent not in roots]
    started = {parent for parent, _ in roots.values()}
    folder = _reservations() if reserved is None else reserved
    for mark in folder.glob("*") if folder.is_dir() else ():
        pid = int(mark.name) if mark.name.isdigit() else 0
        if pid and (proc / str(pid)).is_dir() and pid not in started:
            out.append((pid, "unbake score --permute (starting)"))
        elif not pid or not (proc / str(pid)).is_dir():
            mark.unlink(missing_ok=True)
    return out


def _place(config: Config, proc: Path = Path("/proc")) -> Path:
    """Takes one of CAP places: returns when fewer than CAP runs are alive, saying every 15 s who holds them; a refusal
    after QUEUE_S names them. The returned reservation file is this command's until it ends (starts are serialised by
    a lock, so two starters never both see the same free place)."""
    cap = int(os.environ.get("UNBAKE_PERMUTER_CAP") or CAP)
    began = last = time.monotonic()
    mark = _reservations() / str(os.getpid())
    while True:
        with store.exclusive(config, "permute-start", wait=True):
            held = [h for h in _running(proc) if h[0] != os.getpid()]
            if len(held) < cap:
                mark.parent.mkdir(parents=True, exist_ok=True)
                mark.write_text("")
                return mark
        now = time.monotonic()
        if now - began >= QUEUE_S:
            raise Refusal(
                Finding(
                    "permute.cap",
                    f"{len(held)} permuter runs already hold the {cap} places: "
                    + "; ".join(f"pid {p} {c}" for p, c in held),
                    action="retry when one ends, or stop your own idle run",
                )
            )
        if now == began or now - last >= 15:
            last = now
            effort.say(
                f"permute queued: {len(held)} of {cap} places held by "
                + ", ".join(f"pid {p}" for p, _ in held)
                + f" ({now - began:.0f}s)"
            )
        time.sleep(1.0)


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
    base, defines = _shield(ctx["source"])
    store.write(folder / "base.c", base.encode())
    store.write(folder / "compile.sh", _script(root, ctx["script"], defines).encode())
    (folder / "compile.sh").chmod(0o755)
    weights = "".join(f"perm_{p} = 0\n" for p in _JUNK)
    settings = f'func_name = "{name}"\ncompiler_type = "gcc"\n[weight_overrides]\n{weights}'
    store.write(folder / "settings.toml", settings.encode())
    assemble = [a.replace("{out}", str(folder / "target.o")) for a in ctx["assemble"]]
    done = process.run("permute.target", assemble, root, tmp=work / "tmp")
    if done.exit != 0:
        raise Refusal(Finding("native.exit", f"assembling the target failed: {done.stderr.decode()[-300:]}", unit=name))
    jobs = str(params.get("jobs") or min(8, config.host.workers))
    argv = [
        "/usr/bin/nice",
        "-n",
        "10",
        *_python(permuter),
        permuter,
        "-j",
        jobs,
        "--best-only",
        "--quiet",
        str(folder),
    ]
    mark = _place(config)
    try:
        ran = process.run("permute.run", argv, folder, tmp=work / "tmp", timeout=float(params["permute"]))
    finally:
        mark.unlink(missing_ok=True)
    said = re.search(r"base score = (\d+)", ran.stdout.decode(errors="replace"))
    if said is None and ran.signal != signal.SIGKILL:
        raise Refusal(
            Finding("native.exit", f"permuter said: {ran.stderr.decode()[-1500:] or ran.stdout.decode()[-1500:]}")
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
