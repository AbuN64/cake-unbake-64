"""Class-wise unit order: the text of a merged module with each section class in its own linked order.

A linker places the text, data and read-only data of one module in whatever order its link script lists them, so the
units of a module can be linked text A, B but data B, A. One source file cannot hold that as a concatenation of its
units, but a compiler emits each section in source order, so the file only has to put the code of A before the code
of B and the data of B before the data of A. `arrange` cuts every unit body into chunks (a top-level statement, or the
whole outermost `#if` block it sits in), tells each chunk's class from the symbols the built object of its unit
defines, and emits the chunks without storage (typedefs, prototypes, externs) first, then the storage chunks in an
order that keeps every class in its own unit order and defines a data name before a chunk uses it. A function used
before its definition gets a prototype. What cannot be ordered is reported, never guessed."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Row = dict[str, Any]
CLASSES = ("text", "data", "rodata", "bss")
_IF = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef)\b")
_ENDIF = re.compile(r"^[ \t]*#[ \t]*endif\b")
_WORD = re.compile(r"[A-Za-z_]\w*")


@dataclass(eq=False)
class Chunk:
    unit: str
    at: int  # position of the chunk in its unit
    text: str
    masked: str
    classes: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    heads: dict[str, str] = field(default_factory=dict)  # function name -> its prototype text

    @property
    def cls(self) -> str:
        """The class of the chunk's storage; empty when it holds none."""
        return self.classes[0] if self.classes else ""


def _blocks(masked: str) -> list[tuple[int, int]]:
    """The offsets of the outermost `#if` .. `#endif` blocks of the text."""
    out: list[tuple[int, int]] = []
    depth, start, at = 0, 0, 0
    for line in masked.split("\n"):
        end = at + len(line) + 1
        if _IF.match(line):
            if depth == 0:
                start = at
            depth += 1
        elif _ENDIF.match(line) and depth:
            depth -= 1
            if depth == 0:
                out.append((start, end))
        at = end
    return out


def _fallback(decl: Any, masked: str) -> str:
    """The class of a variable the object does not name: by its text."""
    text = masked[decl.start : decl.end]
    head = text.split("=")[0]
    if "=" not in text:
        return "bss"
    stars = head.rfind("*")
    return "rodata" if re.search(r"\bconst\b", head[stars + 1 :] if stars >= 0 else head) else "data"


def chunks(unit: str, body: str, masked: str, decls: Sequence[Any], classes: Mapping[str, str]) -> list[Chunk]:
    """The chunks of one unit body in order, covering all of it."""
    blocks = _blocks(masked)
    out: list[Chunk] = []
    position, index = 0, 0
    while index < len(decls):
        decl = decls[index]
        end = decl.end + (1 if body[decl.end : decl.end + 1] == "\n" else 0)
        if decl.guard:
            block = next((b for b in blocks if b[0] <= decl.start < b[1]), None)
            if block:
                end = block[1]
        members = [d for d in decls[index:] if d.start < end]
        index += len(members)
        chunk = Chunk(unit, len(out), body[position:end], masked[position:end])
        for d in members:
            _classify(chunk, d, masked, classes)
        out.append(chunk)
        position = end
    if body[position:].strip() and out:
        out[-1].text += body[position:]
        out[-1].masked += masked[position:]
    return out


def _classify(chunk: Chunk, decl: Any, masked: str, classes: Mapping[str, str]) -> None:
    statement = masked[decl.start : decl.end]
    if decl.kind == "def":
        chunk.classes.append("text")
        chunk.defines.append(decl.name)
        chunk.heads[decl.name] = statement[: statement.find("{")].strip() + ";"
    elif decl.kind == "var":
        chunk.classes.append(classes.get(decl.name) or _fallback(decl, masked))
        chunk.defines.append(decl.name)
    elif decl.kind == "proto" and (call := _WORD.match(statement.lstrip())) and call.group(0) == decl.name:
        # a macro call that places a symbol (an included data or code block): the symbols named in its arguments
        for word in _WORD.findall(statement[statement.find("(") :]):
            if word in classes:
                chunk.classes.append(classes[word])
                chunk.defines.append(word)


def arrange(
    parts: Sequence[tuple[str, str, str, Sequence[Any]]],
    orders: Mapping[str, Sequence[str]],
    classes: Mapping[str, Mapping[str, str]],
) -> tuple[str, list[Row]]:
    """The merged body of `parts` ((unit, body, masked body, declarations) in code order) with every class in its
    `orders` unit order (class -> units), and the conflicts left."""
    conflicts: list[Row] = []
    every = [c for unit, body, masked, decls in parts for c in chunks(unit, body, masked, decls, classes.get(unit, {}))]
    for chunk in every:
        if len(set(chunk.classes)) > 1:
            conflicts.append(
                {
                    "kind": "mixed",
                    "name": chunk.defines[0],
                    "detail": f"{chunk.unit}: one block holds {'+'.join(sorted(set(chunk.classes)))}",
                }
            )
    plain = [c for c in every if not c.cls]
    storage = [c for c in every if c.cls]
    rank = {cls: {u: i for i, u in enumerate(order)} for cls, order in orders.items()}
    for cls in {c.cls for c in storage}:
        unplaced = {c.unit for c in storage if c.cls == cls} - rank.get(cls, {}).keys()
        if unplaced:
            rank.setdefault(cls, {}).update({u: len(rank.get(cls, {})) + i for i, u in enumerate(sorted(unplaced))})
    chain: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in sorted(storage, key=lambda c: (rank[c.cls][c.unit], c.at)):
        chain[chunk.cls].append(chunk)
    ordered, forward = _sequence(chain, storage)
    functions = {n: c.heads[n] for c in storage for n in c.heads}
    protos: list[str] = []
    for chunk, names in forward:
        for name in names:
            if name in functions:
                protos.append(functions[name])
            else:
                conflicts.append(
                    {"kind": "forward", "name": name, "detail": f"{chunk.unit}: uses {name} before its definition"}
                )
    out: list[str] = []
    last = ""
    for chunk in plain:
        out.append(chunk.text.strip("\n"))
    if protos:
        out.append("\n".join(dict.fromkeys(protos)))
    for chunk in ordered:
        label = f"/* {chunk.unit.rsplit('/', 1)[-1]} */\n" if chunk.unit != last else ""
        last = chunk.unit
        out.append(label + chunk.text.strip("\n"))
    return "\n\n".join(p for p in out if p.strip()), conflicts


def _sequence(
    chain: Mapping[str, list[Chunk]], storage: Sequence[Chunk]
) -> tuple[list[Chunk], list[tuple[Chunk, list[str]]]]:
    """The storage chunks in one order that keeps each class chain: the next chunk of the class whose head is first in
    the default order (the order of the text chain, then the rest) and whose data names are all defined; when none
    is ready, the first head goes anyway and its early uses are returned."""
    home = {c: i for i, c in enumerate(c for cls in ("text", *CLASSES) for c in chain.get(cls, ()))}
    owner = {n: c for c in storage for n in c.defines if n not in c.heads}
    heads = {cls: list(chunks) for cls, chunks in chain.items()}
    done: set[int] = set()
    out: list[Chunk] = []
    forward: list[tuple[Chunk, list[str]]] = []
    uses = {c: {w for w in _WORD.findall(c.masked) if w in owner and owner[w] is not c} for c in storage}
    while any(heads.values()):
        ready = [h[0] for h in heads.values() if h and all(id(owner[w]) in done for w in uses[h[0]])]
        pick = min(ready or [h[0] for h in heads.values() if h], key=lambda c: home[c])
        if pick not in ready:
            forward.append((pick, sorted(w for w in uses[pick] if id(owner[w]) not in done)))
        heads[pick.cls].pop(0)
        done.add(id(pick))
        out.append(pick)
    for chunk in out:
        later = {n for c in out[out.index(chunk) + 1 :] for n in c.heads}
        used = sorted(w for w in _WORD.findall(chunk.masked) if w in later and w not in chunk.heads)
        if used:
            forward.append((chunk, list(dict.fromkeys(used))))
    return out, forward
