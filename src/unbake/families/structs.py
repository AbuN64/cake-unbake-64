"""Recover shared record identities from the whole landed program.

The evidence graph is directed. An assignment or actual argument can supply an
identity to a local or formal argument; a callee's other callers cannot supply
an identity back to that actual argument. Layout similarity is never an edge.

Analysis uses the real project snapshot and existing content cache. Candidates
are written outside the project, and only submit/land can install sources.
New records need a proven complete storage extent. Unobserved declared padding
or implicit ABI gaps can be filled without changing any real member, sizeof
or alignment. Unknown extents, ambiguous identities, incompatible types and
nonstandard layouts remain conflicts. Every affected unit and header consumer
must have exact native proofs in all its holders before admission.
"""

from __future__ import annotations

import functools
import pickle
import re
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from functools import partial
from io import BytesIO
from itertools import combinations
from types import SimpleNamespace
from typing import Any

from pycparser import CParser, c_ast, c_generator
from pycparser.c_parser import ParseError

from unbake import config as configuration
from unbake import effort, headers, policy, store, types, view
from unbake.contracts import Config, Finding, Json, Refusal, Snapshot, code_identity, digest
from unbake.families import layoutattr, typedebt
from unbake.symbols import by_address


@functools.cache
def _is_view(name: str) -> bool:
    rule = next(r for r in configuration.load_resource("rules.toml")["rule"] if r["id"] == "struct-view-cast")
    bare = name.removeprefix("struct ")
    return bool(re.search(rule["regex"], "(" + bare + " *)")) or typedebt.fake_name(bare)


def _coordinate(node: c_ast.Node | SimpleNamespace, roots: tuple[str, ...]) -> dict[str, Any]:
    coord = node.coord
    path = coord.file if coord else ""
    for root in roots:
        path = path.removeprefix(root.rstrip("/") + "/")
    return {"path": path.removeprefix("./"), "line": coord.line if coord else 0, "column": coord.column if coord else 0}


def _record(
    node: Any, typedefs: Mapping[str, Any], seen: frozenset[str] = frozenset(), pointers: int = 0
) -> str | None:
    if isinstance(node, c_ast.PtrDecl):
        return _record(node.type, typedefs, seen, pointers + 1) if pointers == 0 else None
    if isinstance(node, (c_ast.ArrayDecl, c_ast.TypeDecl)):
        if isinstance(node, c_ast.ArrayDecl) and isinstance(node.type, c_ast.PtrDecl):
            return None
        return _record(node.type, typedefs, seen, pointers)
    if isinstance(node, c_ast.Struct) and node.name:
        return "struct " + node.name
    if isinstance(node, c_ast.Union) and node.name:
        return "union " + node.name
    if isinstance(node, c_ast.IdentifierType) and len(node.names) == 1:
        name = node.names[0]
        if name in typedefs and name not in seen:
            if types.anonymous_record(typedefs[name]) is not None:
                return name
            return _record(typedefs[name], typedefs, seen | {name}, pointers)
    return None


class _Parser(CParser):
    def __init__(self, scope: Mapping[str, bool] | None = None) -> None:
        super().__init__()
        self.scope = scope or {}

    def _parse_translation_unit_or_empty(self) -> c_ast.FileAST:
        self._scope_stack[0].update(self.scope)
        return super()._parse_translation_unit_or_empty()


def _parse(text: str, path: str, scope: Mapping[str, bool] | None = None) -> tuple[c_ast.FileAST, dict[str, bool]]:
    parser = _Parser(scope)
    try:
        ast = parser.parse(headers.normal(text), filename=path)
    except ParseError as error:
        raise headers.parse_refusal(error, path) from error
    return ast, parser._scope_stack[0]


def _prefix(text: str, path: str, roots: Sequence[str]) -> tuple[str, str]:
    current, marker, offset = path, 0, 0
    start: int | None = None
    source_markers: list[tuple[int, int]] = []
    for line in text.splitlines(keepends=True):
        found = re.match(r'^# (\d+) "([^"\n]*)"', line)
        if found:
            current, marker = found[2], offset
            for root in roots:
                current = current.removeprefix(root.rstrip("/") + "/")
            current = current.removeprefix("./")
            if current == path:
                source_markers.append((offset, offset + len(line)))
        elif line.strip():
            if start is None:
                start = marker
            if current == path:
                prefix = text[start:marker]
                for lo, hi in reversed(source_markers):
                    if start <= lo < marker:
                        prefix = prefix[: lo - start] + prefix[hi - start :]
                return prefix, text[marker:]
        offset += len(line)
    return text[start or 0 :], ""


def _prepare_key(item: tuple[Any, ...]) -> str:
    snapshot, unit, versions, recipe, real_root = item
    return digest(
        (
            snapshot.config.project.build,
            unit.path,
            real_root,
            recipe.digest,
            [(v, snapshot.config.project.version_macros[v], view.closure(snapshot, unit, v)) for v in versions],
        )
    )


def _prepare_job(item: tuple[Any, ...]) -> list[tuple[str, str | None]]:
    snapshot, unit, versions, recipe, real_root = item
    config, raw = snapshot.config, snapshot.peek(unit.path) or b""
    roots = (str(config.project.root), real_root)
    results: list[tuple[str, str | None]] = []
    prepared: dict[str, str | None] = {}
    for version in versions:
        try:
            seen = view.get(snapshot, unit, version, recipe, lines=False)
            dependencies: list[tuple[str, str]] = []
            for path_, pin in seen.dependencies:
                for root in roots:
                    path_ = path_.removeprefix(root.rstrip("/") + "/")
                dependencies.append((path_, pin))
            key = digest((code_identity(_prepare_job), seen.text, unit.path, roots, raw, dependencies))
            if key not in prepared:
                prefix, body = _prefix(seen.text, unit.path, roots)
                prefix_key = digest((code_identity(_prepare_job), prefix))
                store.cached(config, "recovery.prefix.text", prefix_key, partial(str.encode, prefix))
                payload: tuple[Any, ...] = (body, unit.path, roots, raw, dependencies, prefix_key, None)
                store.put(config, "recovery.input", key, pickle.dumps(payload, protocol=5))
                prepared[key] = prefix_key
        except Exception as error:
            error = _failure(error, unit.path)
            key = digest((code_identity(_prepare_job), unit.path, raw, error.findings))
            payload = ("", unit.path, roots, raw, (), None, error)
            store.put(config, "recovery.input", key, pickle.dumps(payload, protocol=5))
            prepared[key] = None
        results.append((key, prepared[key]))
    return results


def _stored(item: tuple[Any, ...], kind: str) -> bytes:
    job, key = item
    value = store.get(job[0].config, kind, key)
    if value is None:
        _prepare_job(job)
        value = store.get(job[0].config, kind, key)
    assert value is not None  # preparing the job stores it
    return value


def _index(ast: c_ast.FileAST) -> tuple[Any, ...]:
    typedefs, records, duplicates = types._record_index(ast)
    fields: dict[tuple[str, str | None], Any] = {}
    owners: dict[str, Any] = {}
    anonymous: dict[Any, Any] = {}

    class Fields(c_ast.NodeVisitor):
        def visit_Struct(self, node: c_ast.Struct) -> None:
            if node.decls:
                if node.name:
                    name = "struct " + node.name
                    owners[name] = node
                    fields.update(
                        {(name, d.name): d for d in node.decls if not (d.name or "").startswith(layoutattr.MARKER)}
                    )
                else:
                    anonymous[node] = node.decls
            self.generic_visit(node)

    Fields().visit(ast)
    return ast.ext, typedefs, records, duplicates, fields, owners, anonymous


def _merge(indexes: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    nodes: list[Any] = []
    typedefs: dict[str, Any] = {}
    records: dict[str, Any] = {}
    duplicates: set[str] = set()
    fields: dict[Any, Any] = {}
    owners: dict[str, Any] = {}
    anonymous: dict[Any, Any] = {}
    for ext, td, rec, dup, fs, own, anon in indexes:
        nodes.extend(ext)
        typedefs.update(td)
        duplicates.update(dup | records.keys() & rec.keys())
        records.update(rec)
        fields.update(fs)
        owners.update(own)
        anonymous.update(anon)
    return nodes, typedefs, records, duplicates, fields, owners, anonymous


def _fragment(config: Config, text: str, scope: Mapping[str, bool]) -> tuple[str, Any]:
    words = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
    reached = {name: scope[name] for name in sorted(words & scope.keys())}
    key = digest((code_identity(_fragment), text, reached))

    def produce() -> bytes:
        with effort.stage("recovery.fragment.parse"):
            rewritten, tainted = layoutattr.rewrite(text)
            ast, after = _parse(rewritten, "", reached)
        with effort.stage("recovery.fragment.index"):
            indexed = _index(ast)
        metadata = {n: v for n, v in after.items() if reached.get(n) != v}, tainted
        return pickle.dumps(metadata, protocol=5) + pickle.dumps(indexed, protocol=5)

    value = pickle.loads(store.cached(config, "recovery.fragment", key, produce))
    return key, value


def _prefix_job(item: tuple[Any, ...]) -> None:
    job, key = item
    config = job[0].config
    parsed: tuple[list[str], dict[str, bool], set[str]] | Refusal | None
    try:
        text = _stored(item, "recovery.prefix.text").decode()
        parts: list[str] = []
        scope: dict[str, bool] = {}
        tainted: set[str] = set()
        pending = ""
        for chunk in re.split(r'(?m)(?=^# \d+ ")', text):
            pending += chunk
            if not re.sub(r"(?m)^#.*", "", pending).strip():
                continue
            try:
                part, (added, marked) = _fragment(config, pending, scope)
            except Refusal:
                continue
            parts.append(part)
            scope.update(added)
            tainted.update(marked)
            pending = ""
        parsed = (parts, scope, tainted) if not re.sub(r"(?m)^#.*", "", pending).strip() else None
    except Exception as error:
        parsed = _failure(error, job[1].path)
    store.put(config, "recovery.prefix", key, pickle.dumps(parsed, protocol=5))


# The last prefix this process scanned: its fragments' decoded indexes and their merged index.
_HEAD: list[Any] = [None, {}, None]


def _decode(job: tuple[Any, ...], prefix_key: str, part: str) -> Any:
    config = job[0].config
    value = store.get(config, "recovery.fragment", part)
    if value is None:
        _prefix_job((job, prefix_key))
        value = store.get(config, "recovery.fragment", part)
    assert value is not None  # the prefix job stores every fragment
    stream = BytesIO(value)
    pickle.load(stream)
    return pickle.load(stream)


def _head(job: tuple[Any, ...], prefix_key: str, parts: Sequence[str]) -> tuple[Any, ...]:
    """The merged index of a prefix, decoding only the fragments the previous prefix did not hold."""
    if _HEAD[0] == prefix_key:
        merged: tuple[Any, ...] = _HEAD[2]
        return merged
    held = _HEAD[1]
    decoded = {part: held[part] if part in held else _decode(job, prefix_key, part) for part in parts}
    merged = _merge([decoded[part] for part in parts])
    _HEAD[:] = [prefix_key, decoded, merged]
    return merged


def _scan_batch(item: tuple[Any, ...]) -> list[dict[str, Any]]:
    """The evidence of every source that shares one prefix: the prefix is loaded and merged once."""
    members, _ = item
    return [_scan_job(member) for member in members]


def _scan_job(item: tuple[Any, ...]) -> dict[str, Any]:
    job, _ = item
    config = job[0].config
    path, raw = job[1].path, job[0].peek(job[1].path) or b""
    try:
        text, path, roots, raw, dependencies, prefix_key, error = pickle.loads(_stored(item, "recovery.input"))
        if error:
            raise error

        def parsed_prefix() -> Any:
            value = store.get(config, "recovery.prefix", prefix_key)
            if value is None:
                _prefix_job((job, prefix_key))
                value = store.get(config, "recovery.prefix", prefix_key)
            assert value is not None  # the prefix job stores it
            return pickle.loads(value)

        prefix = parsed_prefix()
        if isinstance(prefix, Refusal):
            raise prefix
        if prefix is None:
            text = _stored((job, prefix_key), "recovery.prefix.text").decode() + text
            return source_evidence(text, path, roots, raw, dependencies)
        parts, scope, marked = prefix

        with effort.stage("recovery.source.headers"):
            merged = _head(job, prefix_key, parts)
        with effort.stage("recovery.source.parse"):
            text, tainted = layoutattr.rewrite(text)
            body, _ = _parse(text, path, scope)
        with effort.stage("recovery.source.evidence"):
            return _evidence(body, marked | tainted, path, roots, raw, dependencies, merged=merged)
    except Exception as error:
        error = _failure(error, path)
        code = policy.scopes(raw.decode(errors="replace"))["code"]
        code = re.sub(r"\bstruct\s+(?=[A-Za-z_]\w*\s*\*)", lambda m: " " * len(m[0]), code)
        pattern = next(
            r["regex"] for r in configuration.load_resource("rules.toml")["rule"] if r["id"] == "struct-view-cast"
        )
        casts = []
        for match in re.finditer(pattern, code):
            view_name = re.search(r"\bfunc_\w+", match[0])
            name = view_name[0] if view_name else ""
            casts.append(
                {
                    "origin": None,
                    "view": name,
                    "operand": "",
                    "function": "",
                    "path": path,
                    "line": code.count("\n", 0, match.start()) + 1,
                    "column": match.start() - code.rfind("\n", 0, match.start()),
                    "conflict": "; ".join(f.reason for f in error.findings),
                }
            )
        return {
            "path": path,
            "casts": casts,
            "declarations": [],
            "flows": [],
            "accesses": [],
            "source_digest": digest(raw),
            "records": {},
            "conflicts": [{"reason": f.reason, "path": f.path, "line": f.line} for f in error.findings],
        }


def _failure(error: Exception, path: str) -> Refusal:
    findings = (
        error.findings
        if isinstance(error, Refusal)
        else [Finding("headers.parse", reason=f"{type(error).__name__}: {error}")]
    )
    return Refusal(*(Finding("headers.parse", reason=f.reason, path=path, line=f.line) for f in findings))


def source_evidence(
    text: str, path: str, roots: tuple[str, ...] = (), raw: bytes = b"", dependencies: Sequence[tuple[str, str]] = ()
) -> dict[str, Any]:
    """Content-only scan, shared by real pool jobs and mocked tests."""
    text, tainted = layoutattr.rewrite(text)
    ast, _ = _parse(text, path)
    return _evidence(ast, tainted, path, roots, raw, dependencies)


def _evidence(
    ast: c_ast.FileAST,
    tainted: Collection[str],
    path: str,
    roots: tuple[str, ...],
    raw: bytes,
    dependencies: Sequence[tuple[str, str]],
    head: Sequence[Any] = (),
    merged: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    with effort.stage("recovery.evidence.index"):
        nodes, typedefs, records, duplicates, fields, owners, anonymous = _merge(
            [merged if merged is not None else _merge(head), _index(ast)] if merged or head else [_index(ast)]
        )
        for name, typ in typedefs.items():
            if isinstance(typ, c_ast.TypeDecl) and typ.type in anonymous:
                owners[name] = typ.type
                fields.update(
                    {(name, d.name): d for d in anonymous[typ.type] if not (d.name or "").startswith(layoutattr.MARKER)}
                )
        fields = {key: (decl, "field:" + key[0] + ":" + (key[1] or "")) for key, decl in fields.items()}
        owners = {name: _coordinate(node, roots) for name, node in owners.items()}
        layout_index = typedefs, records, duplicates
    generator = c_generator.CGenerator()
    declarations: list[dict[str, Any]] = []
    casts: list[dict[str, Any]] = []
    flows: list[tuple[str, str, str]] = []
    accesses: list[dict[str, Any]] = []
    conflicts: list[Any] = []
    globals_: dict[str, Any] = {}
    functions: dict[str, list[Any]] = {}
    returns: dict[str, Any] = {}
    object_uses: list[dict[str, Any]] = []
    extents: list[dict[str, Any]] = []
    char_type = c_ast.TypeDecl("bytes", [], None, c_ast.IdentifierType(["char"]))

    def storage(typ: Any) -> int | None:
        name = "struct __unbake_extent"
        decl = c_ast.Decl("bytes", [], None, [], [], typ, None, None)
        probe = c_ast.Struct("__unbake_extent", [decl])
        index = typedefs, {**layout_index[1], name: probe}, duplicates
        return types.record_layouts(ast, [name], tainted, index=index).get(name, {}).get("size")

    used_records: set[str] = set()
    handled_casts: set[int] = set()
    words = set(re.findall(r"\b\w+\b", raw.decode(errors="replace"))) if raw else None
    internal = {
        decl.name
        for node in nodes
        for decl in [node.decl if isinstance(node, c_ast.FuncDef) else node]
        if isinstance(decl, c_ast.Decl) and "static" in decl.storage
    }

    def identity(name: str) -> str:
        return path + ":" + name if name in internal else name

    def declared(
        key: str, node: Any, kind: str, owner: str = "", index: int | None = None
    ) -> tuple[Any, dict[str, Any]]:
        record = _record(node.type, typedefs)
        canonical = headers.canonical(node, typedefs) if node.type is not None else ""
        row = {
            "key": key,
            "kind": kind,
            "name": getattr(node, "name", None) or "",
            "type": canonical,
            "record": record,
            "owner": owner,
            "index": index,
            **_coordinate(node, roots),
        }
        if kind == "global":
            row["defined"] = "extern" not in node.storage
            row["external"] = node.name not in internal
        declarations.append(row)
        if isinstance(node.type, c_ast.ArrayDecl) and re.match(r"(?:const )?(?:unsigned |signed )?char \[", canonical):
            size = storage(node.type)
            if size:
                row["type"] = re.sub(r"\[[^]]*\]", "[" + str(size) + "]", canonical, count=1)
                extents.append({"origin": key, "size": size, "kind": "declaration"})
        if record:
            used_records.add(record)
        return node.type, row

    for node in nodes:
        decl = node.decl if isinstance(node, c_ast.FuncDef) else node
        if (
            not isinstance(node, c_ast.FuncDef)
            and isinstance(decl, c_ast.Decl)
            and words is not None
            and decl.name
            and decl.name not in words
            and _coordinate(decl, roots)["path"] != path
        ):
            continue
        if isinstance(decl, c_ast.Decl) and isinstance(decl.type, c_ast.FuncDecl):
            params = decl.type.args.params if decl.type.args else ()
            old = {p.name: p for p in (node.param_decls or ())} if isinstance(node, c_ast.FuncDef) else {}
            parameters = []
            for param in params:
                if isinstance(param, c_ast.EllipsisParam):
                    continue
                if isinstance(param, c_ast.ID):
                    implicit = c_ast.TypeDecl(param.name, [], None, c_ast.IdentifierType(["int"]))
                    param = old.get(
                        param.name,
                        SimpleNamespace(
                            name=param.name,
                            coord=param.coord,
                            type=implicit if isinstance(node, c_ast.FuncDef) else None,
                        ),
                    )
                parameters.append(param)
            assert decl.name is not None  # a function declaration has a name
            functions[decl.name] = parameters
            returns[decl.name] = decl.type.type
            if isinstance(decl.type.type, c_ast.PtrDecl):
                returned = SimpleNamespace(type=decl.type.type, name=decl.name, coord=decl.coord)
                declared("return:" + identity(decl.name), returned, "return", decl.name)
            for index, param in enumerate(functions[decl.name]):
                declared(f"param:{identity(decl.name)}:{index}", param, "param", decl.name, index)
        elif isinstance(decl, c_ast.Decl) and decl.name:
            globals_[decl.name] = declared("global:" + identity(decl.name), decl, "global")
    used_records.clear()

    class Body(c_ast.NodeVisitor):
        def __init__(self, function: str) -> None:
            self.function = function
            self.scopes: list[dict[str, tuple[Any, str]]] = [{}]
            self.parents: list[Any] = []
            for index, param in enumerate(functions[function]):
                if getattr(param, "name", None):
                    self.scopes[0][param.name] = (param.type, f"param:{identity(function)}:{index}")

        def visit(self, node: c_ast.Node) -> Any:
            self.parents.append(node)
            try:
                return super().visit(node)
            finally:
                self.parents.pop()

        def visit_ID(self, node: c_ast.ID) -> None:
            parent = self.parents[-2] if len(self.parents) > 1 else None
            if isinstance(parent, c_ast.StructRef) and parent.field is node:
                return
            _, key = self.lookup(node.name)
            if key and key.startswith("global:"):
                within_view = any(
                    isinstance(p, c_ast.Cast) and _is_view(_record(p.to_type.type, typedefs) or "")
                    for p in self.parents[:-1]
                )
                address = isinstance(parent, c_ast.UnaryOp) and parent.op == "&"
                context = "view" if within_view else "address" if address else "other"
                object_uses.append({"origin": key, "context": context, **_coordinate(node, roots)})

        def lookup(self, name: str) -> tuple[Any, str | None]:
            for scope in reversed(self.scopes):
                if name in scope:
                    return scope[name]
            if name in globals_:
                node, row = globals_[name]
                return node, row["key"]
            return None, None

        def expression(self, node: Any) -> tuple[Any, str | None]:
            if isinstance(node, c_ast.ID):
                return self.lookup(node.name)
            if isinstance(node, c_ast.Cast):
                typ, key = self.expression(node.expr)
                spelling = generator.visit(node.to_type)
                if re.fullmatch(r"(?:const\s+)?(?:void|char|signed char|unsigned char|s8|u8)\s*\*", spelling):
                    return typ, key
                return node.to_type.type, key
            if isinstance(node, c_ast.Assignment) and node.op == "=":
                return self.expression(node.rvalue)
            if isinstance(node, c_ast.UnaryOp) and node.op == "&":
                typ, key = self.expression(node.expr)
                return c_ast.PtrDecl([], typ), key
            if isinstance(node, (c_ast.ArrayRef, c_ast.UnaryOp)):
                if isinstance(node, c_ast.UnaryOp) and node.op != "*":
                    return None, None
                typ, key = self.expression(node.name if isinstance(node, c_ast.ArrayRef) else node.expr)
                if isinstance(typ, (c_ast.PtrDecl, c_ast.ArrayDecl)) and key:
                    element = typ.type
                    key = ("indirect:" if isinstance(element, c_ast.PtrDecl) else "table:") + key
                    return element, key
            if isinstance(node, c_ast.BinaryOp) and node.op == "+":
                for base, offset in ((node.left, node.right), (node.right, node.left)):
                    typ, key = self.expression(base)
                    if isinstance(base, c_ast.Cast):
                        typ = base.to_type.type
                    if key and isinstance(typ, (c_ast.PtrDecl, c_ast.ArrayDecl)):
                        coord = _coordinate(node, roots)
                        origin = f"element:{key}:{path}:{coord['line']}:{coord['column']}"
                        stride = offset.right if isinstance(offset, c_ast.BinaryOp) and offset.op == "*" else None
                        size = storage(c_ast.ArrayDecl(char_type, stride, [])) if stride else None
                        if size:
                            extents.append(
                                {
                                    "origin": origin,
                                    "size": size,
                                    "kind": "stride",
                                    "function": self.function,
                                    "base": key,
                                }
                            )
                        return typ, origin
            if isinstance(node, c_ast.StructRef) and isinstance(node.field, c_ast.ID):
                parent, parent_origin = self.expression(node.name)
                record = _record(parent, typedefs) if parent else None
                field = fields.get((record, node.field.name)) if parent_origin else None
                if field:
                    decl, key = field
                    if not any(d["key"] == key for d in declarations):
                        declared(key, decl, "field")
                    return decl.type, key
            if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID):
                return_type = returns.get(node.name.name)
                if isinstance(return_type, c_ast.PtrDecl):
                    coord = _coordinate(node, roots)
                    key = f"call:{path}:{coord['line']}:{coord['column']}:{node.name.name}"
                    if not any(d["key"] == key for d in declarations):
                        returned = SimpleNamespace(type=return_type, name=node.name.name, coord=node.coord)
                        declared(key, returned, "call", node.name.name)
                        args = node.args.exprs if node.args else []
                        size = None
                        if node.name.name == "malloc" and len(args) == 1:
                            size = storage(c_ast.ArrayDecl(char_type, args[0], []))
                        elif node.name.name == "calloc" and len(args) == 2:
                            size = storage(c_ast.ArrayDecl(char_type, c_ast.BinaryOp("*", *args), []))
                        if size:
                            extents.append({"origin": key, "size": size, "kind": "allocation"})
                        elif node.name.name not in ("malloc", "calloc"):
                            flows.append(("return:" + identity(node.name.name), key, "declared return"))
                    return return_type, key
            return None, None

        def visit_Compound(self, node: c_ast.Compound) -> None:
            self.scopes.append({})
            for child in node.block_items or ():
                self.visit(child)
            self.scopes.pop()

        def visit_For(self, node: c_ast.For) -> None:
            self.scopes.append({})
            self.generic_visit(node)
            self.scopes.pop()

        def visit_Decl(self, node: c_ast.Decl) -> None:
            if node.name and not isinstance(node.type, c_ast.FuncDecl):
                coord = _coordinate(node, roots)
                key = f"local:{path}:{self.function}:{coord['line']}:{coord['column']}:{node.name}"
                declared(key, node, "local", self.function)
                self.scopes[-1][node.name] = node.type, key
                if node.init is not None:
                    _, origin = self.expression(node.init)
                    if origin:
                        flows.append((origin, key, "initializer"))
            self.generic_visit(node)

        def visit_Return(self, node: c_ast.Return) -> None:
            if node.expr is not None and isinstance(returns.get(self.function), c_ast.PtrDecl):
                _, origin = self.expression(node.expr)
                if origin:
                    flows.append((origin, "return:" + identity(self.function), "return"))
            self.generic_visit(node)

        def visit_Assignment(self, node: c_ast.Assignment) -> None:
            _, target = self.expression(node.lvalue)
            _, origin = self.expression(node.rvalue)
            if target and origin and node.op == "=":
                flows.append((origin, target, "assignment"))
                typ, _ = self.expression(node.lvalue)
                if typ and not isinstance(typ, c_ast.PtrDecl) and _record(typ, typedefs):
                    size = storage(typ)
                    if size:
                        extents.append({"origin": target, "size": size, "kind": "struct copy"})
            self.generic_visit(node)

        def visit_FuncCall(self, node: c_ast.FuncCall) -> None:
            if isinstance(node.name, c_ast.ID) and node.name.name in ("memcpy", "memmove", "bcopy") and node.args:
                args = node.args.exprs
                if len(args) == 3:
                    size = storage(c_ast.ArrayDecl(char_type, args[2], []))
                    for arg in args[:2]:
                        _, origin = self.expression(arg)
                        if origin and size:
                            extents.append({"origin": origin, "size": size, "kind": "copy bound"})
            if isinstance(node.name, c_ast.ID) and node.name.name in functions and node.args:
                for index, arg in enumerate(node.args.exprs):
                    _, origin = self.expression(arg)
                    params = functions[node.name.name]
                    if origin and index < len(params) and isinstance(params[index].type, c_ast.PtrDecl):
                        flows.append((origin, f"param:{identity(node.name.name)}:{index}", "argument"))
            self.generic_visit(node)

        def visit_StructRef(self, node: c_ast.StructRef) -> None:
            typ, origin = self.expression(node.name)
            record = _record(typ, typedefs) if typ else None
            name = node.field.name if isinstance(node.field, c_ast.ID) else ""
            if isinstance(node.name, c_ast.Cast):
                view_record = _record(node.name.to_type.type, typedefs)
                source_type, origin = self.expression(node.name.expr)
                param_cast = bool(
                    view_record
                    and not _is_view(view_record)
                    and (origin or "").startswith("param:")
                    and _opaque(source_type)
                )
                if view_record and (_is_view(view_record) or param_cast):
                    source_record = view_record if param_cast else None
                    if source_type is not None and not param_cast:
                        source_record = _record(source_type, typedefs)
                    if source_record:
                        used_records.add(source_record)
                    row = {
                        "origin": origin,
                        "view": view_record,
                        "field": name,
                        "operand": generator.visit(node.name.expr),
                        "function": self.function,
                        "source_record": source_record,
                        **_coordinate(node.name, roots),
                    }
                    base = node.name.expr
                    while isinstance(base, c_ast.Cast):
                        base = base.expr
                    row["real_operand"] = generator.visit(base)
                    if not origin:
                        row["conflict"] = "no proven scoped origin: " + row["operand"]
                    casts.append(row)
                    handled_casts.add(id(node.name))
                    used_records.add(view_record)
            if record:
                used_records.add(record)
                accesses.append({"origin": origin, "record": record, "field": name, **_coordinate(node, roots)})
            self.generic_visit(node)

        def visit_Cast(self, node: c_ast.Cast) -> None:
            record = _record(node.to_type.type, typedefs)
            if record and _is_view(record) and id(node) not in handled_casts:
                _, origin = self.expression(node.expr)
                casts.append(
                    {
                        "origin": origin,
                        "view": record,
                        "operand": generator.visit(node.expr),
                        "function": self.function,
                        "conflict": "view cast without direct field access",
                        **_coordinate(node, roots),
                    }
                )
                used_records.add(record)
            self.generic_visit(node)

    for node in ast.ext:
        if isinstance(node, c_ast.FuncDef) and _coordinate(node.decl, roots)["path"] == path:
            Body(node.decl.name).visit(node.body)
    needed = {u["origin"] for u in casts if u["origin"]} | {a["origin"] for a in accesses if a["origin"]}
    needed.update(k for a, b, _ in flows for k in (a, b))
    own_functions = {
        n.decl.name for n in ast.ext if isinstance(n, c_ast.FuncDef) and _coordinate(n.decl, roots)["path"] == path
    }
    declarations = [
        d
        for d in declarations
        if d["key"] in needed
        or d["owner"] in own_functions
        or (d["kind"] == "global" and d["defined"] and d["path"] == path)
    ]
    used_records.update(d["record"] for d in declarations if d["record"])
    with effort.stage("recovery.evidence.layouts"):
        records = types.record_layouts(ast, used_records, tainted, index=layout_index)
    conflicts.extend(
        {"path": path, "reason": f"{n}: {records[n]['conflict']}"}
        for n in sorted(used_records)
        if records.get(n, {}).get("conflict", "").startswith("unsupported layout")
    )
    for row in casts:
        if "conflict" in row:
            continue
        layout_ = records.get(row["view"], {})
        field = next((f for f in layout_.get("fields", ()) if f["name"] == row["field"]), None)
        if field:
            row.update(
                offset=field["offset"],
                width=field["width"],
                type=field["type"],
                fields=field.get("fields", []),
                declaration=field["declaration"],
                alignment=field["alignment"],
            )
        else:
            row["conflict"] = layout_.get("conflict", "view field has no complete declaration")

    # Standalone casts are retained as exact unresolved evidence, rather than silently skipped.
    raw_code = policy.scopes(raw.decode(errors="replace"))["code"] if raw else ""
    pattern = next(
        r["regex"] for r in configuration.load_resource("rules.toml")["rule"] if r["id"] == "struct-view-cast"
    )
    count = len(re.findall(pattern, raw_code))
    if count > len(casts):
        conflicts.append({"path": path, "reason": f"{count - len(casts)} view casts without a direct field access"})
    return {
        "path": path,
        "casts": casts,
        "declarations": declarations,
        "flows": sorted(set(flows)),
        "accesses": accesses,
        "records": {n: records[n] for n in used_records if n in records},
        "conflicts": conflicts,
        "source_digest": digest(raw),
        "dependencies": dependencies,
        "record_owners": {n: owners[n] for n in used_records if n in owners},
        "object_uses": object_uses,
        "extents": list({digest(e): e for e in extents}.values()),
    }


def _aggregate(field: dict[str, Any]) -> bool:
    """Arrays and by-value records; a pointer (to anything) or a scalar is one access."""
    kind = field["type"].rstrip()
    return kind.endswith("]") or (kind.startswith(("struct", "union")) and not kind.endswith("*"))


def _element(field: dict[str, Any]) -> str:
    return re.sub(r"(?:\s*\[[^\[\]]*\])+$", "", field["type"].rstrip())


_HUB = 10  # records reaching one formal at which it is a helper, not an identity


def _agree(left: dict[str, Any], right: dict[str, Any], shift: int = 0) -> bool:
    """Two layouts describe one object when their fields share storage by offset and width and never straddle it.

    An aggregate field may cover several scalars of the other layout. Scalars at one offset and width must
    have one C type (any two pointers count as one access). At least one scalar must coincide, so an empty
    or disjoint record never merges.
    """
    if "conflict" in left or "conflict" in right:
        return False
    shared = False
    for a in left.get("fields", ()):
        for b in ({**f, "offset": f["offset"] + shift} for f in right.get("fields", ())):
            if a["offset"] >= b["offset"] + b["width"] or b["offset"] >= a["offset"] + a["width"]:
                continue
            if (a["offset"], a["width"]) == (b["offset"], b["width"]):
                if not (_aggregate(a) or _aggregate(b)):
                    opaque = "void *" in (a["type"], b["type"])  # untyped storage: a conflict row, not a split
                    if a["type"] == b["type"] or opaque or (a["type"].endswith("*") and b["type"].endswith("*")):
                        shared = True
                    elif not (a.get("arm") or b.get("arm")):  # a union arm is one alternative: others may differ
                        return False
                shared = shared or _element(a) == _element(b)
            elif not (_aggregate(a) or _aggregate(b) or a.get("arm") or b.get("arm")):
                return False
    return shared or shift != 0  # records laid side by side in one object share no scalar


def _overlap(left: dict[str, Any], right: dict[str, Any], shift: int) -> float:
    """The share of the smaller layout's fields the other places at one offset and width."""
    slots = {(f["offset"] + shift, f["width"]) for f in right.get("fields", ())}
    both = sum((f["offset"], f["width"]) in slots for f in left.get("fields", ()))
    return both / max(1, min(len(left.get("fields", ())), len(right.get("fields", ()))))


def _loser(left: dict[str, Any], right: dict[str, Any], shift: int, seen: Mapping[int, int], origin: int) -> int | None:
    """0 or 1: the side whose scalar width at one offset the assembly access width contradicts."""
    for a in left.get("fields", ()):
        for b in right.get("fields", ()):
            measured = seen.get(a["offset"] + origin)
            if (
                a["offset"] == b["offset"] + shift
                and a["width"] != b["width"]
                and not (_aggregate(a) or _aggregate(b))
                and measured in (a["width"], b["width"])
            ):
                return int(measured == a["width"])
    return None


def _classes(
    groups: Mapping[str, set[str]],
    layouts: Mapping[str, Any],
    places: Mapping[str, Mapping[str, int]] | None = None,
    asm: Mapping[str, Mapping[int, int]] | None = None,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    """Classes of records that reach one origin (or one object, at `places` offsets) and agree on storage.

    Each class maps its members to their byte offset from the lowest. Agreement is not transitive: two classes
    join only when every member of one agrees with every member of the other, so one sparse record never bridges
    records that disagree with each other. A record whose C width the assembly contradicts is a conflict row
    and never merges.
    """
    places, asm = places or {}, asm or {}
    agreed: dict[tuple[str, str, int], bool] = {}

    def agree(x: str, y: str, shift: int) -> bool:
        if (x, y, shift) not in agreed:
            agreed[x, y, shift] = x in layouts and y in layouts and _agree(layouts[x], layouts[y], shift)
        return agreed[x, y, shift]

    member: dict[str, set[str]] = {}
    pos: dict[str, int] = {}
    lost: set[str] = set()
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    queued: set[tuple[str, str, int]] = set()
    pairs: list[tuple[Any, ...]] = []
    rows = []
    support: Counter[tuple[str, str, int]] = Counter()  # groups reaching a pair; a hub group counts as a weak one
    for key, group in sorted(groups.items(), key=lambda row: (sorted(row[1]), row[0])):
        frame = places.get(key, {})
        stamp = (frozenset(group), tuple(sorted(frame.items())))
        if len(group) < 2 or stamp in seen:
            continue
        seen.add(stamp)
        rows.append((key, group, frame))
        for left, right in combinations(sorted(group), 2):
            support[left, right, frame.get(right, 0) - frame.get(left, 0)] += 2 if len(group) < _HUB else 1
    for key, group, frame in rows:
        for left, right in combinations(sorted(group), 2):
            gap = frame.get(right, 0) - frame.get(left, 0)
            if support[left, right, gap] < 2:  # only a hub (many records through one helper) links them: no identity
                continue
            if asm.get(key) or (left, right, gap) not in queued:
                queued.add((left, right, gap))
                strength = -_overlap(layouts[left], layouts[right], gap) if agree(left, right, gap) else 0
                pairs.append((strength, len(pairs), left, right, key, gap))
    for _, _, left, right, key, gap in sorted(pairs):  # strongest evidence merges first, across every origin
        frame = places.get(key, {})
        if lost & {left, right}:
            continue
        a, b = member.setdefault(left, {left}), member.setdefault(right, {right})
        pos.setdefault(left, 0)
        pos.setdefault(right, 0)
        if a is b:
            continue
        k = pos[left] + gap - pos[right]
        if not all(agree(x, y, pos[y] + k - pos[x]) for x in a for y in b):
            # split: the parts of both classes that agree with the other side's anchor join; a class that would be
            # left with one member is not broken up (that member would only lose its place)
            part = {x for x in a if agree(x, right, pos[right] + k - pos[x])}
            other = {y for y in b if agree(left, y, pos[y] + k - pos[left])}
            if (
                left in part
                and right in other
                and len(a - part) != 1
                and len(b - other) != 1
                and all(agree(x, y, pos[y] + k - pos[x]) for x in part for y in other)
            ):
                for rest, kept in ((a - part, part), (b - other, other)):
                    for n in rest:
                        member[n] = rest
                    for n in kept:
                        member[n] = kept
                a, b = member[left], member[right]
        if not all(agree(x, y, pos[y] + k - pos[x]) for x in a for y in b):
            pair = (left, right)
            if (
                left in layouts
                and right in layouts
                and (who := _loser(layouts[left], layouts[right], gap, asm.get(key, {}), frame.get(left, 0)))
                is not None
            ):
                lost.add(pair[who])
                conflicts.append({"reason": "c width contradicts assembly", "record": pair[who], "with": pair[1 - who]})
            continue
        for n in b:
            pos[n] += k
            member[n] = a
        a |= b
    classes = []
    for members in (m for m in {id(m): m for m in member.values()}.values() if len(m) > 1):
        low = min(pos[n] for n in members)
        classes.append({n: pos[n] - low for n in sorted(members)})
    return classes, conflicts


def _identities(anchors: Mapping[str, set[str]], layouts: Mapping[str, Any]) -> dict[str, str]:
    """Map records that reach one origin together and agree on storage to one representative (the fullest layout)."""
    result: dict[str, str] = {}
    for members in _classes(anchors, layouts)[0]:
        rep = min(members, key=lambda n: (-len(layouts.get(n, {}).get("fields", ())), n))
        result.update({n: rep for n in members if n != rep})
    return result


def _alias_anchors(objects: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Symbols one object spans: an access inside symbol S through a base register holding symbol B joins S and B.
    The object starts at its lowest symbol; `widths` are the assembly access widths by offset from it."""
    found: dict[frozenset[str], dict[str, Any]] = {}
    for table in objects.values():
        sets: dict[str, set[str]] = {}
        for a, b in table.get("pairs", ()):
            joined = sets.get(a, {a}) | sets.get(b, {b})
            sets.update(dict.fromkeys(joined, joined))
        for group in {id(g): g for g in sets.values()}.values():
            at = sorted((table["symbols"][n], n) for n in group if n in table["symbols"])
            if len(at) < 2:
                continue
            stop = bisect_right(table["order"], at[-1][0])
            top = table["order"][stop] if stop < len(table["order"]) else at[-1][0] + 1
            reach = table["addresses"][
                bisect_right(table["addresses"], at[0][0] - 1) : bisect_right(table["addresses"], top - 1)
            ]
            found.setdefault(
                frozenset(group),
                {
                    "members": {n: a - at[0][0] for a, n in at},
                    "widths": {a - at[0][0]: table["accesses"][a]["width"] for a in reach},
                },
            )
    return [found[k] for k in sorted(found, key=sorted)]


def _covers(
    symbol: str, size: int, objects: Mapping[str, Any], friends: Collection[str], named: Mapping[str, Any]
) -> list[str]:
    """Separate symbols whose address lies inside the first `size` bytes of `symbol`."""
    for version, table in objects.items():
        if symbol in table["symbols"]:
            start = table["symbols"][symbol]
            lo, hi = bisect_right(table["order"], start), bisect_right(table["order"], start + size - 1)
            return [n for a in table["order"][lo:hi] if (n := named[version][a][0]) not in friends]
    return []


def _map(layout: Mapping[str, Any], delta: int, flat: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each member field to the canonical field at its storage (name, clean C type); a type the canonical cannot
    stand in for is marked `conflict` so a rewrite skips it."""
    rows = []
    for f in layout.get("fields", ()):
        lo, hi = f["offset"] + delta, f["offset"] + delta + f["width"]
        at = next((c for c in flat if c["offset"] <= lo and hi <= c["offset"] + c["width"]), None)
        row: dict[str, Any] = {"offset": f["offset"], "width": f["width"], "name": f["name"], "type": f["type"]}
        if at:
            row.update(canonical=at["name"], canonical_type=at["type"], canonical_offset=at["offset"])
            if (at["offset"], at["width"]) == (lo, f["width"]) and f["type"] != at["type"]:
                row["conflict"] = "type differs: " + f["type"] + " vs " + at["type"]
        else:
            row["conflict"] = "no canonical field"
        rows.append(row)
    return rows


def _canonical(members: Mapping[str, int], owners: Mapping[str, Any], uses: Mapping[str, int]) -> str:
    """The shared-header record with the most consumers; per-function names never win."""

    def rank(name: str) -> tuple[Any, ...]:
        path = next((o["path"] for o in owners.get(name, ()) if o["path"].startswith("include/")), "")
        bare = name.removeprefix("struct ")
        invented = typedebt.fake_name(bare) or re.search(r"_u\d+$", bare)
        return (bool(invented), not path, not path.startswith("include/shared/"), -uses.get(name, 0), name)

    return min(members, key=rank)


def _flat(canonical: str, members: Mapping[str, int], layouts: Mapping[str, Any], uses: Mapping[str, int]) -> list[Any]:
    """One flat non-union layout: the canonical's fields, then the others' scalars, then their aggregates."""
    rows = [
        (n != canonical, _aggregate(f), -uses.get(n, 0), n, {**f, "offset": f["offset"] + d})
        for n, d in members.items()
        for f in layouts.get(n, {}).get("fields", ())
    ]
    taken: list[tuple[int, int]] = []
    out = []
    for *_, f in sorted(rows, key=lambda r: (*r[:4], r[4]["offset"])):
        lo, hi = f["offset"], f["offset"] + f["width"]
        if "union" not in f["type"] and not any(lo < end and start < hi for start, end in taken):
            taken.append((lo, hi))
            out.append({k: f[k] for k in ("name", "offset", "width", "type", "declaration") if k in f})
    return sorted(out, key=lambda f: f["offset"])


def families(evidence: list[dict[str, Any]], objects: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return graph((evidence, objects or {}, ()))[0]


def graph(
    item: tuple[list[dict[str, Any]], Mapping[str, Any], Collection[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propagate actual-to-formal evidence, then group by a proven shared record or one origin.

    Also returns the identity table's inputs; `_components` splits them for the pool and `table` builds the rows."""
    evidence, objects, reads = item
    declarations, successors, predecessors = defaultdict(list), defaultdict(set), defaultdict(set)
    layouts: dict[str, Any] = {}
    collisions: defaultdict[Any, list[Any]] = defaultdict(list)
    owners: defaultdict[Any, list[Any]] = defaultdict(list)
    paths, casts = {}, []
    for row in evidence:
        paths[row["path"]] = row
        casts.extend(row["casts"])
        for decl in row["declarations"]:
            if decl["path"].startswith(("src/", "include/")):
                declarations[decl["key"]].append(decl)
        for a, b, _ in row["flows"]:
            successors[a].add(b)
            predecessors[b].add(a)
        for name, record in row["records"].items():
            if name in layouts and layouts[name] != record:
                collisions[name].append(
                    {
                        "path": row["path"],
                        "reason": "record declarations disagree",
                        "left": layouts[name],
                        "right": record,
                    }
                )
            layouts[name] = record
        for name, owner in row.get("record_owners", {}).items():
            if owner not in owners[name]:
                owners[name].append(owner)
    casts = list({digest(cast): cast for cast in casts}.values())
    anchors = {
        k: {d["record"] for d in ds if d["record"] and not _is_view(d["record"])} for k, ds in declarations.items()
    }
    # Concrete declared interfaces impose a type constraint on arguments/assignments.
    # This is a separate backward pass seeded ONLY by declarations. An identity
    # inherited from a callee's other caller never propagates back to an actual.
    contracts = {k: set(v) for k, v in anchors.items()}
    todo = deque(sorted(contracts))
    while todo:
        target = todo.popleft()
        for origin in predecessors[target]:
            before = len(contracts.setdefault(origin, set()))
            contracts[origin].update(contracts[target])
            if len(contracts[origin]) != before:
                todo.append(origin)
    for key, values in contracts.items():
        anchors.setdefault(key, set()).update(values)
    keys = declarations.keys() | {c["origin"] for c in casts if c["origin"]}
    roots = {k: {k} if not predecessors[k] and not k.startswith("local:") else set() for k in keys}
    for key in keys:
        anchors.setdefault(key, set())
    for k in successors:
        anchors.setdefault(k, set())
        roots.setdefault(k, {k} if not predecessors[k] and not k.startswith("local:") else set())
    for cast in casts:
        if cast.get("source_record") and not _is_view(cast["source_record"]) and cast["origin"]:
            anchors.setdefault(cast["origin"], set()).add(cast["source_record"])
    queue, queued = deque(sorted(anchors)), set(anchors)
    while queue:
        a = queue.popleft()
        queued.discard(a)
        for b in successors[a]:
            old = len(anchors.setdefault(b, set())), len(roots.setdefault(b, set()))
            anchors[b].update(anchors[a])
            roots[b].update(roots[a])
            if old != (len(anchors[b]), len(roots[b])) and b not in queued:
                queue.append(b)
                queued.add(b)
    # A formal no callee body reads a field through (a forwarder or opaque helper) anchors nothing.
    touched = (
        set(reads)
        | {a["origin"] for row in evidence for a in row["accesses"] if a["origin"]}
        | {c["origin"] for c in casts if c["origin"]}
    )
    todo = deque(touched)
    while todo:
        for parent in predecessors[todo.popleft()] - touched:
            touched.add(parent)
            todo.append(parent)

    def helper(key: str) -> bool:
        return key.startswith("param:") and key not in touched

    # Agreeing records resolve only an origin where several identities meet; elsewhere families stay apart.
    merged = _identities({k: v for k, v in anchors.items() if not helper(k)}, layouts)
    table = _table_spec(evidence, declarations, successors, helper, layouts, owners, objects)
    grouped = defaultdict(list)
    for cast in casts:
        origin = cast["origin"]
        inherited = anchors.get(origin, set())
        if len(inherited) > 1 and len(collapsed := {merged.get(v, v) for v in inherited}) == 1:
            inherited = collapsed
        origins = roots.get(origin, set())
        if len(inherited) == 1:
            family = "record:" + next(iter(inherited))
        elif len(inherited) > 1:
            family = "conflict:" + (origin or cast["operand"])
        elif len(origins) == 1:
            family = "origin:" + next(iter(origins))
        else:
            family = "unknown:" + (origin or cast["path"] + ":" + cast["operand"])
        grouped[family].append({**cast, "anchors": sorted(inherited), "roots": sorted(origins)})
    # Evidence rows by origin and record, so a family reads only its own (position order kept across the lookups).
    by_origin: dict[str, defaultdict[Any, list[tuple[int, Any]]]] = {
        kind: defaultdict(list) for kind in ("extents", "object_uses", "accesses")
    }
    by_record: defaultdict[Any, list[tuple[int, Any]]] = defaultdict(list)
    for n, (kind, a) in enumerate((k, a) for row in evidence for k in by_origin for a in row.get(k, ())):
        by_origin[kind][a["origin"]].append((n, a))
        if kind == "accesses":
            by_record[a["record"]].append((n, a))

    def rows_at(kind: str, origins: Collection[Any], records: Collection[Any] = ()) -> list[Any]:
        found = {n: a for o in origins for n, a in by_origin[kind].get(o, ())}
        found.update({n: a for r in records for n, a in by_record.get(r, ())})
        return [found[n] for n in sorted(found)]

    extents_at = partial(rows_at, "extents")
    object_uses_at = partial(rows_at, "object_uses")
    accesses_at = partial(rows_at, "accesses")
    jobs = []
    for identity, uses in sorted(grouped.items()):
        nodes = {u["origin"] for u in uses if u["origin"]}
        # Include every declaration/access of the typed family, not only the source with a cast.
        selected = next(iter(uses))["anchors"]
        if len(selected) == 1:
            nodes.update(k for k in declarations if anchors.get(k) == {selected[0]})
        needed = set(nodes)
        walk = list(nodes)
        while walk:
            for parent in predecessors[walk.pop()] - needed:
                needed.add(parent)
                walk.append(parent)
        involved = {u["path"] for u in uses}
        involved.update(d["path"] for k in needed for d in declarations[k])
        reached = {d["record"] for k in needed for d in declarations[k] if d["record"]}
        reached.update(u["view"] for u in uses)
        reached.update(u["source_record"] for u in uses if u.get("source_record"))
        reached.update(merged[r] for r in list(reached) if r in merged)
        declarations_ = [d for k in sorted(needed) for d in declarations[k]]
        # Deduplicate header declarations appearing in many preprocessed units.
        declarations_ = list({digest(d): d for d in declarations_}.values())
        common_headers = None
        for p in {u["path"] for u in uses}:
            dependencies = {n for n, _ in paths[p].get("dependencies", ()) if n.startswith("include/")}
            common_headers = dependencies if common_headers is None else common_headers & dependencies
        header_owners = sorted({d["path"] for d in declarations_ if d["path"].startswith("include/")})
        jobs.append(
            {
                "id": digest(identity)[:16],
                "identity": identity,
                "casts": uses,
                "declarations": declarations_,
                "extents": extents_at(needed),
                "records": {n: layouts[n] for n in sorted(reached) if n in layouts},
                "identities": {n: merged[n] for n in sorted(reached) if n in merged},
                "conflicts": [c for n in reached for c in collisions[n]]
                + [c for p in involved if p in paths for c in paths[p]["conflicts"]],
                "sources": {p: paths[p]["source_digest"] for p in sorted(involved) if p in paths},
                "record_owners": {n: owners[n] for n in sorted(reached) if n in owners},
                "new_owner": next(iter(header_owners or sorted(common_headers or ())), None),
                "object_uses": object_uses_at(nodes),
                "accesses": [a for a in accesses_at(nodes, selected) if not _is_view(a["record"])],
            }
        )
    return jobs, table


def _table_spec(
    evidence: Sequence[dict[str, Any]],
    declarations: Mapping[str, list[dict[str, Any]]],
    successors: Mapping[str, set[str]],
    helper: Callable[[str], bool],
    layouts: Mapping[str, Any],
    owners: Mapping[str, Any],
    objects: Mapping[str, Any],
) -> dict[str, Any]:
    """The identity table's inputs: the origin and alias groups, and what the rows need once the classes are known."""
    reached = {k: {d["record"] for d in ds if d["record"]} for k, ds in declarations.items()}
    todo = deque(reached)
    while todo:
        a = todo.popleft()
        for b in successors[a]:
            if not reached[a] <= reached.setdefault(b, set()):
                reached[b] |= reached[a]
                todo.append(b)
    groups = {k: v for k, v in reached.items() if not helper(k)}
    places: dict[str, dict[str, int]] = {}
    asm: dict[str, dict[int, int]] = {}
    symbols: defaultdict[str, set[str]] = defaultdict(set)
    for ds in declarations.values():
        for d in ds:
            if d["kind"] == "global" and d["record"] and d.get("external"):
                symbols[d["record"]].add(d["name"])
    for alias in _alias_anchors(objects):
        key = "alias:" + min(alias["members"])
        places[key] = {}
        for symbol, delta in alias["members"].items():
            for record in {r for r, names in symbols.items() if symbol in names}:
                places[key][record] = min(delta, places[key].get(record, delta))
        groups[key] = set(places[key])
        asm[key] = alias["widths"]
    uses = Counter(d["record"] for ds in declarations.values() for d in ds if d["record"])
    uses.update(a["record"] for row in evidence for a in row["accesses"])
    return {
        "groups": groups,
        "places": places,
        "asm": asm,
        "symbols": dict(symbols),
        "uses": uses,
        "layouts": layouts,
        "owners": owners,
    }


def components(spec: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    """Independent `_classes` items: groups that share no record never meet, so each component runs on its own core."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in spec["groups"].values():
        for record in group:
            parent[find(record)] = find(next(iter(group)))
    parts: defaultdict[str, list[str]] = defaultdict(list)
    for key, group in spec["groups"].items():
        if len(group) > 1:
            parts[find(next(iter(group)))].append(key)
    return [
        (
            {k: spec["groups"][k] for k in keys},
            {r: spec["layouts"][r] for k in keys for r in spec["groups"][k] if r in spec["layouts"]},
            {k: spec["places"][k] for k in keys if k in spec["places"]},
            {k: spec["asm"][k] for k in keys if k in spec["asm"]},
        )
        for keys in parts.values()
    ]


def classes_job(item: tuple[Any, ...]) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    return _classes(*item)


def identity_table(
    spec: Mapping[str, Any], parts: Sequence[tuple[Any, ...]], objects: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The identity table: every class of records (views included) reaching one origin or spanning one object."""
    layouts, owners, symbols, uses = spec["layouts"], spec["owners"], defaultdict(set, spec["symbols"]), spec["uses"]
    classes = [c for cs, _ in parts for c in cs]
    lost = [c for _, ls in parts for c in ls]
    named = {version: by_address(table["symbols"]) for version, table in objects.items()}
    rows = []
    for members in sorted(classes, key=lambda m: sorted(m)):
        canonical = _canonical(members, owners, uses)
        home = next((o["path"] for o in owners.get(canonical, ()) if o["path"].startswith("include/")), None)
        found = [c for c in lost if c["with"] in members]
        flat = _flat(canonical, members, layouts, uses)
        friends = {s for r in members for s in symbols[r]}
        for record in members:
            for symbol in sorted(symbols[record]):
                size = layouts.get(record, {}).get("size") or 0
                if covered := _covers(symbol, size, objects, friends, named):
                    found.append({"reason": "field covers a separate symbol", "record": record, "covers": covered})
        rows.append(
            {
                "canonical": canonical,
                "owner_header": home,
                "members": [
                    {
                        "name": n,
                        "path": next((o["path"] for o in owners.get(n, ())), None),
                        "delta": d,
                        "map": _map(layouts.get(n, {}), d, flat),
                    }
                    for n, d in members.items()
                ],
                "globals": [{"symbol": s, "offset": d} for n, d in members.items() for s in sorted(symbols[n])],
                "fields": flat,
                "size": max((layouts.get(n, {}).get("size", 0) + d for n, d in members.items()), default=0),
                "conflicts": found,
            }
        )
    return rows


def _derive_key(job: dict[str, Any]) -> str:
    return digest(job)


def _opaque(typ: Any) -> bool:
    """A `void *` or integer declaration: storage of no record identity."""
    pointer = isinstance(typ, c_ast.PtrDecl)
    inner = typ.type if pointer else typ
    names = getattr(getattr(inner, "type", None), "names", None) or []
    return bool(names) and (names == ["void"] if pointer else set(names) <= typedebt._INTEGER)


def _byte_extent(spelling: str) -> int | None:
    match = re.fullmatch(r"(?:const\s+)?(?:unsigned |signed )?char\s*\[\s*(\d+)\s*\]", spelling)
    return int(match[1]) if match else None


def _new_record(job: dict[str, Any], conflicts: list[Any]) -> tuple[str | None, dict[str, Any], list[Any]]:
    """Require complete storage evidence for every possible root, never just touched bytes."""
    roots = {o for use in job["casts"] for o in use["roots"]}
    declarations = job["declarations"]
    evidence = list(job.get("extents", ()))
    evidence.extend(
        {"origin": d["key"], "size": _byte_extent(d["type"]), "kind": "declaration"}
        for d in declarations
        if _byte_extent(d["type"])
    )
    observed = ("declaration", "allocation", "symbol", "struct copy", "assembly stride")
    sizes = {root: {e["size"] for e in evidence if e["origin"] == root and e["kind"] in observed} for root in roots}
    strides = {e["origin"] for e in evidence if e["kind"] == "stride"}
    extents = set().union(*sizes.values()) if sizes else set()
    open_extent = bool(roots) and not any(sizes.values()) and not strides
    if open_extent:
        shaped = [u for u in job["casts"] if "conflict" not in u]
        align = max((u["alignment"] for u in shaped), default=1)
        end = max((u["offset"] + u["width"] for u in shaped), default=0)
        extent = -(-end // align) * align
    elif not roots or any(len(v) != 1 for v in sizes.values()) or len(extents) != 1:
        conflicts.append(
            {
                "reason": "new record has no proven complete object extent",
                "origins": sorted(roots),
                "declarations": declarations,
                "evidence": evidence,
                "detail": "observed extent is unproven"
                if strides
                else "observers disagree or are incomplete; source.pad-view forbids padding substitutes",
            }
        )
        return None, {}, []
    else:
        extent = next(iter(extents))
    if any(e["size"] > extent for e in evidence if e["kind"] == "copy bound"):
        conflicts.append(
            {"reason": "copy exceeds proven complete object extent", "extent": extent, "evidence": evidence}
        )
    objects = [
        d
        for d in declarations
        if _byte_extent(d["type"]) or re.fullmatch(r"(?:const )?(?:unsigned |signed )?char \[\]", d["type"])
    ]
    if objects and any(u["context"] == "other" for u in job.get("object_uses", ())):
        conflicts.append(
            {
                "reason": "byte object has uses outside a view or address operation",
                "uses": [u for u in job["object_uses"] if u["context"] == "other"],
            }
        )
    if any(d.get("defined") for d in objects):
        conflicts.append(
            {
                "reason": "byte object has a C definition whose initializer migration is unproven",
                "declarations": [d for d in objects if d.get("defined")],
            }
        )
    if not job.get("new_owner"):
        conflicts.append({"reason": "new record has no existing shared header owner"})
    fields: dict[str, Any] = {}
    for use in job["casts"]:
        if "conflict" in use:
            continue
        at = use["offset"]
        if at in fields and (fields[at]["width"], fields[at]["type"]) != (use["width"], use["type"]):
            conflicts.append(
                {
                    "reason": "same object offset has incompatible field types",
                    "offset": at,
                    "left": fields[at],
                    "right": use,
                }
            )
        elif at not in fields:
            fields[at] = {
                "offset": at,
                "width": use["width"],
                "type": use["type"],
                "alignment": use["alignment"],
                "name": use["field"],
                "declaration": use["declaration"],
            }
    ordered = sorted(fields.values(), key=lambda f: f["offset"])
    end, alignment = 0, 1
    names = set()
    for field in ordered:
        alignment = max(alignment, field["alignment"])
        if field["offset"] < end or field["offset"] % field["alignment"] or field["offset"] + field["width"] > extent:
            conflicts.append(
                {
                    "reason": "field overlaps storage, violates alignment or exceeds the object extent",
                    "field": field,
                    "extent": extent,
                    "previous_end": end,
                }
            )
        if field["name"] in names:
            conflicts.append({"reason": "one field name denotes two offsets", "field": field})
        if any(_is_view(word) for word in re.findall(r"\b\w+\b", field["type"])):
            conflicts.append({"reason": "new field points through an unresolved view type", "field": field})
        names.add(field["name"])
        end = field["offset"] + field["width"]
    if extent % alignment:
        conflicts.append(
            {
                "reason": "record alignment disagrees with declared byte-object size",
                "extent": extent,
                "alignment": alignment,
            }
        )
    for version, accesses_ in job.get("global_assembly", {}).items():
        for access in accesses_:
            base = access.get("base")
            offset = base[1] if base and base[0] == next(iter(roots)).removeprefix("global:") else None
            if offset is None or access["width"] == 0 or access["how"] == "taken":
                continue
            if offset >= extent:
                continue
            if not any(f["offset"] == offset and f["width"] == access["width"] for f in ordered):
                conflicts.append(
                    {
                        "reason": "assembly object access has no consistent C field evidence",
                        "version": version,
                        "offset": offset,
                        "width": access["width"],
                        "evidence": access,
                    }
                )
    target = "struct Recovered_" + job["id"]
    record: dict[str, Any] = {"size": extent, "alignment": alignment, "fields": ordered}
    changes = [{"kind": "new", "record": target, "owner": job.get("new_owner"), "layout": record}]
    if open_extent:
        record["open"] = changes[0]["open"] = True
    return target, record, changes


def _extension(
    job: dict[str, Any], target: str, record: dict[str, Any], use: dict[str, Any], conflicts: list[Any]
) -> dict[str, Any] | None:
    """Use only implicit ABI padding, preserving all existing members, offsets, alignment and sizeof."""
    start, stop = use["offset"], use["offset"] + use["width"]
    proven = {"declaration", "allocation", "symbol", "struct copy", "assembly stride", "copy bound"}
    open_ = bool(record.get("open")) or (
        target.startswith("struct Recovered_") and not any(e["kind"] in proven for e in job.get("extents", ()))
    )
    if start < 0 or (stop > record.get("size", 0) and not open_) or start % use["alignment"]:
        return None
    overlapping = [f for f in record.get("fields", ()) if f["offset"] < stop and start < f["offset"] + f["width"]]
    padding = None
    if overlapping:
        if len(overlapping) != 1:
            return None
        padding = overlapping[0]
        if (
            not re.match(r"^_*pad(?:ding)?(?:[0-9A-Fa-f_]|$)", padding["name"])
            or _byte_extent(padding["type"]) != padding["width"]
            or start < padding["offset"]
            or stop > padding["offset"] + padding["width"]
        ):
            return None
        observed = [a for a in job["accesses"] if a["record"] == target and a["field"] == padding["name"]]
        observed += [
            u for u in job["casts"] if u.get("offset") == padding["offset"] and u.get("type") == padding["type"]
        ]
        if observed:
            conflicts.append(
                {"reason": "declared padding is observed as a byte array", "padding": padding, "accesses": observed}
            )
            return None
    if use["alignment"] > record["alignment"] and not open_:
        return None
    owners = job.get("record_owners", {}).get(target, ())
    if len(owners) != 1 or not owners[0]["path"].startswith("include/") or not target.startswith("struct "):
        conflicts.append(
            {"reason": "record extension needs one shared definition owner", "record": target, "owners": list(owners)}
        )
        return None
    if any(f["name"] == use["field"] for f in record["fields"]):
        return None
    if padding:
        record["fields"].remove(padding)
        prefix = start - padding["offset"]
        suffix = padding["offset"] + padding["width"] - stop
        for offset, width, name in (
            (padding["offset"], prefix, padding["name"]),
            (stop, suffix, padding["name"] + "_tail"),
        ):
            if width:
                record["fields"].append(
                    {
                        "name": name,
                        "offset": offset,
                        "width": width,
                        "alignment": 1,
                        "type": f"unsigned char [{width}]",
                        "declaration": f"unsigned char {name}[{width}];",
                        "fields": [],
                    }
                )
    if open_:
        record["alignment"] = max(record["alignment"], use["alignment"])
        record["size"] = -(-max(record["size"], stop) // record["alignment"]) * record["alignment"]
    field = {k: use[k] for k in ("offset", "width", "type", "alignment", "declaration")}
    return {**field, "name": use["field"]}


def _validate_assembly(job: dict[str, Any], conflicts: list[Any]) -> None:
    """Keep infer's memory widths with each C access and reject contradictory observed widths.

    Missing ROM flow is reported as absent evidence, never fabricated; exact admission
    remains mandatory for every holder and consumer before any source can be landed.
    """
    for use in job["casts"]:
        if "offset" not in use:
            continue
        for version, functions in job.get("assembly", {}).items():
            if use["function"] not in functions and version in job.get("holders", {}).get(use["function"], ()):
                conflicts.append(
                    {
                        "reason": "cast function has no assembly evidence",
                        "path": use["path"],
                        "function": use["function"],
                        "version": version,
                    }
                )
            prefix = f"param:{use['function']}:"
            if (use.get("origin") or "").startswith(prefix) and "[" not in use["type"]:
                index = int(use["origin"].removeprefix(prefix))
                widths = {
                    w
                    for k, offset, w in functions.get(use["function"], {}).get("args", ())
                    if k == index and offset == use["offset"]
                }

                def leaf_widths(fields: Iterable[Json]) -> set[int]:
                    return {
                        w
                        for f in fields
                        if f["offset"] == 0
                        for w in (leaf_widths(f["fields"]) if f.get("fields") else {f["width"]})
                    }

                expected = leaf_widths(use.get("fields", ())) or {use["width"]}
                if widths and not widths <= expected:
                    conflicts.append(
                        {
                            "reason": "assembly argument access disagrees with field width",
                            "path": use["path"],
                            "line": use["line"],
                            "version": version,
                            "offset": use["offset"],
                            "c_width": use["width"],
                            "asm_widths": sorted(widths),
                        }
                    )
            # infer may stop tracking an argument after aliasing/indirect calls. An untracked
            # access cannot establish a new field; existing declarations still establish layout.


def derive(job: dict[str, Any]) -> dict[str, Any]:
    """A complete field map or exact conflicts. No candidate is emitted on any conflict."""
    conflicts = list(job["conflicts"])
    uses = job["casts"]
    anchors = sorted({a for u in uses for a in u["anchors"]})
    changes: list[Any] = []
    if len(anchors) > 1:
        conflicts.append(
            {
                "reason": "multiple real record identities reach this origin",
                "anchors": anchors,
                "origins": sorted({o for u in uses for o in u["roots"]}),
            }
        )
    target = anchors[0] if len(anchors) == 1 else None
    record = deepcopy(job["records"].get(target, {}))
    if not anchors:
        target, record, changes = _new_record(job, conflicts)
    if target and ("conflict" in record or not record):
        conflicts.append({"reason": record.get("conflict", "shared record is incomplete"), "record": target})
    mapping = []
    for use in uses:
        if "conflict" in use:
            conflicts.append({k: use[k] for k in ("path", "line", "operand", "conflict") if k in use})
            continue
        possible = [
            f
            for f in record.get("fields", ())
            if f["offset"] == use["offset"] and f["width"] == use["width"] and f["type"] == use["type"]
        ]
        if not possible and anchors and target:
            field = _extension(job, target, record, use, conflicts)
            if field:
                if not changes:
                    changes.append(
                        {
                            "kind": "extend",
                            "record": target,
                            "owner": job["record_owners"][target][0]["path"],
                            "layout": dict(record),
                        }
                    )
                    record = {**record, "fields": list(record["fields"])}
                record["fields"].append(field)
                changes[0]["layout"] = record
                possible = [field]
        if len(possible) != 1:
            conflicts.append(
                {
                    "reason": "no unique shared field with identical offset, width and C type",
                    "path": use["path"],
                    "line": use["line"],
                    "record": target,
                    "offset": use["offset"],
                    "width": use["width"],
                    "type": use["type"],
                    "existing": [
                        f
                        for f in record.get("fields", ())
                        if f["offset"] < use["offset"] + use["width"] and use["offset"] < f["offset"] + f["width"]
                    ],
                }
            )
        else:
            mapping.append(
                {
                    **use,
                    "real_field": possible[0]["name"],
                    "record": target,
                    "typed_cast": (use.get("origin") or "").startswith(("call:", "element:", "indirect:", "table:")),
                }
            )
    # A scalar ABI value, array or function pointer is not silently made a record pointer.
    declarations = []
    for decl in job["declarations"]:
        if decl["kind"] == "call":
            continue
        if (
            changes
            and changes[0]["kind"] == "new"
            and re.fullmatch(r"(?:const )?(?:unsigned |signed )?char \[\d*\]", decl["type"])
        ):
            declarations.append({**decl, "object": True})
            continue
        if decl["record"] == target:
            continue
        merged = decl["record"] and job.get("identities", {}).get(decl["record"]) == target
        if decl["record"] and not _is_view(decl["record"]) and not merged:
            conflicts.append({"reason": "existing declaration has a different real record identity", **decl})
        elif (
            "*" not in decl["type"]
            or decl["type"].count("*") != 1
            or "[" in decl["type"]
            or "(" in decl["type"]
            or "volatile" in decl["type"]
        ):
            conflicts.append({"reason": "origin is not a simple pointer declaration", **decl})
        else:
            declarations.append(decl)
    _validate_assembly(job, conflicts)
    return {
        "id": job["id"],
        "identity": job["identity"],
        "record": target,
        "files": sorted({u["path"] for u in uses}),
        "casts": len(uses),
        "conflicts": conflicts,
        "mapping": [] if conflicts else mapping,
        "declarations": [] if conflicts else declarations,
        "sources": job["sources"],
        "record_changes": [] if conflicts else changes,
        "record_owners": job.get("record_owners", {}),
    }


def _record_body(record: dict[str, Any]) -> str:
    fields = sorted(record["fields"], key=lambda f: f["offset"])
    lines, end = [], 0
    names = {f["name"] for f in fields}
    for field in fields:
        if field["offset"] > end:
            name = f"padding_{end:X}"
            if name in names:
                raise ValueError("generated padding name conflicts: " + name)
            lines.append(f"    unsigned char {name}[{field['offset'] - end}];")
        lines.append("    " + field["declaration"])
        end = field["offset"] + field["width"]
    if record["size"] > end:
        name = f"padding_{end:X}"
        if name in names:
            raise ValueError("generated padding name conflicts: " + name)
        lines.append(f"    unsigned char {name}[{record['size'] - end}];")
    return "\n".join(lines)


def _balanced(code: str, start: int, opening: str = "(", closing: str = ")") -> int:
    depth = 0
    for pos in range(start, len(code)):
        depth += (code[pos] == opening) - (code[pos] == closing)
        if depth == 0:
            return pos + 1
    raise ValueError("unbalanced operand")


def _operand(code: str, start: int) -> tuple[int, int]:
    pos = start
    while pos < len(code) and code[pos].isspace():
        pos += 1
    if pos < len(code) and code[pos] == "(":
        return pos, _balanced(code, pos)
    match = re.match(r"&?\s*[A-Za-z_]\w*(?:(?:\s*->\s*|\s*\.\s*)[A-Za-z_]\w*)*", code[pos:])
    if not match:
        raise ValueError("unsupported operand")
    end = pos + match.end()
    while (tail := re.match(r"\s*[\[(]", code[end:])) is not None:
        opening = code[end + tail.end() - 1]
        end = _balanced(code, end + tail.end() - 1, opening, "]" if opening == "[" else ")")
    return pos, end


def _tokens(expression: str) -> str:
    # Only proven ID, address, array and member operands reach the rewriter.
    return re.sub(r"[\s()]", "", expression)


def rewrite(result: dict[str, Any], contents: dict[str, bytes]) -> dict[str, bytes]:
    """Coordinate-bound source edits, with stale text and overlapping edits refused."""
    edits = defaultdict(list)
    for use in result["mapping"]:
        path = use["path"]
        code = policy.scopes(contents[path].decode())["code"]
        pattern = r"\(\s*(?:struct\s+)?" + re.escape(use["view"].removeprefix("struct ")) + r"\s*\*\s*\)"
        candidates = []
        for match in re.finditer(pattern, code):
            if code.count("\n", 0, match.start()) + 1 != use["line"]:
                continue
            a, b = _operand(code, match.end())
            if _tokens(code[a:b]) == _tokens(use["operand"]):
                candidates.append((match.start(), b))
        if not candidates:
            raise ValueError(f"{path}:{use['line']}: cast coordinate does not match source")
        line_start = code.rfind("\n", 0, candidates[0][0]) + 1
        wanted = line_start + use["column"] - 1
        start, stop = min(candidates, key=lambda p: abs(p[0] - wanted))
        # The field is outside any enclosing parentheses around the cast expression.
        field = re.match(r"\s*\)*\s*->\s*(\w+)", code[stop:])
        if field is None or field[1] != use["field"]:
            raise ValueError(f"{path}:{use['line']}: field coordinate does not match source")
        field_start = stop + field.start(1)
        operand = use.get("real_operand", use["operand"])
        object_ = any(d.get("object") and d["key"] == use["origin"] for d in result["declarations"])
        if object_ and not operand.startswith("&"):
            operand = "&" + operand
        if use.get("typed_cast"):
            operand = "(" + result["record"] + " *)(" + operand + ")"
        edits[path].append((start, stop, "(" + operand + ")"))
        edits[path].append((field_start, field_start + len(field[1]), use["real_field"]))
    for decl in result["declarations"]:
        path = decl["path"]
        if path not in contents:
            raise ValueError("missing declaration owner: " + path)
        code = policy.scopes(contents[path].decode())["code"]
        if not decl["name"] and decl["kind"] == "param":
            matches = list(re.finditer(r"\b" + re.escape(decl["owner"]) + r"\s*\(", code))
            matches = [m for m in matches if code.count("\n", 0, m.start()) + 1 == decl["line"]]
            if len(matches) != 1:
                raise ValueError(f"{path}:{decl['line']}: unnamed parameter owner is ambiguous")
            a, b = matches[0].end(), _balanced(code, matches[0].end() - 1) - 1
            parts = list(headers.parts(code[a:b]))
            index = decl["index"]
            if index >= len(parts) or not re.fullmatch(r"\s*(?:const\s+)?(?:struct\s+)?\w+\s*\*\s*", parts[index]):
                raise ValueError(f"{path}:{decl['line']}: unnamed parameter shape is not proven")
            start = a + sum(len(p) + 1 for p in parts[:index])
            qualifier = "const " if "const" in parts[index].split() else ""
            edits[path].append((start, start + len(parts[index]), qualifier + result["record"] + " *"))
            continue
        if not decl["name"]:
            raise ValueError(f"{path}:{decl['line']}: declaration has no name")
        starts = [0, *(m.end() for m in re.finditer("\n", code))]
        start = starts[decl["line"] - 1]
        stop = code.find("\n", start)
        stop = len(code) if stop < 0 else stop
        qualifier = "const " if re.search(r"\bconst\b", decl["type"]) else ""
        if decl.get("object"):
            pattern = r"(?:\b(?:const|unsigned|signed)\s+)*\bchar\s+" + re.escape(decl["name"]) + r"\s*\[[^]\n]*\]"
            replacement = qualifier + result["record"] + " " + decl["name"]
        else:
            pattern = (
                r"(?:\b(?:const|volatile|struct|unsigned|signed)\s+)*\b\w+\s*\*\s*" + re.escape(decl["name"]) + r"\b"
            )
            replacement = qualifier + result["record"] + " *" + decl["name"]
        matches = list(re.finditer(pattern, code[start:stop]))
        if len(matches) != 1:
            raise ValueError(f"{path}:{decl['line']}: pointer declaration coordinate is ambiguous")
        match = matches[0]
        edits[path].append((start + match.start(), start + match.end(), replacement))
    for change in result.get("record_changes", ()):
        path = change["owner"]
        text = contents[path].decode()
        if change["kind"] == "new":
            declaration = change["record"] + " {\n" + _record_body(change["layout"]) + "\n};\n\n"
            # Install before any migrated extern in this owner; type names must be visible first.
            owner_lines = [d["line"] for d in result["declarations"] if d["path"] == path]
            if owner_lines:
                starts = [0, *(m.end() for m in re.finditer("\n", text))]
                position = starts[min(owner_lines) - 1]
            else:
                endifs = list(re.finditer(r"^\s*#\s*endif\b", text, re.M))
                position = endifs[-1].start() if endifs else len(text)
            edits[path].append((position, position, declaration))
        else:
            tag = change["record"].removeprefix("struct ")
            found = []
            for a, b, _ in headers.statements(text):
                opened = re.search(r"\bstruct\s+" + re.escape(tag) + r"\s*\{", text[a:b])
                if opened:
                    found.append((a + opened.end(), a + text[a:b].rfind("}")))
            if len(found) != 1:
                raise ValueError("shared record definition is ambiguous: " + change["record"])
            a, b = found[0]
            edits[path].append((a, b, "\n" + _record_body(change["layout"]) + "\n"))
    writes = {}
    for path, changes in edits.items():
        text = contents[path].decode()
        previous = len(text) + 1
        for start, stop, replacement in sorted(set(changes), reverse=True):
            if stop > previous:
                raise ValueError(f"{path}: overlapping source edits")
            text = text[:start] + replacement + text[stop:]
            previous = start
        if text.encode() != contents[path]:
            writes[path] = text.encode()
    return writes


def _rewrite_job(item: tuple[Any, ...]) -> dict[str, Any]:
    result, contents = item
    try:
        return {"writes": rewrite(result, contents), "conflicts": []}
    except ValueError as error:
        return {"writes": {}, "conflicts": [{"reason": str(error)}]}


def _type_job(snapshot: Snapshot) -> Json:
    return types.load(snapshot)


def _type_key(snapshot: Snapshot) -> str:
    return digest(snapshot.read("types.toml"))


def object_evidence(item: tuple[Any, ...]) -> dict[str, Any]:
    symbols, usages, _ = item
    names = by_address(symbols)
    order = sorted(names)
    accesses: dict[str, Any] = {}
    pairs: set[tuple[str, str]] = set()
    for use in usages.values():
        for address, width, _, how, base in use["accesses"]:
            at = bisect_right(order, base) - 1 if base is not None else -1
            inside = bisect_right(order, address) - 1
            if at >= 0 and inside >= 0 and names[order[inside]][0] != names[order[at]][0]:
                pairs.add((names[order[inside]][0], names[order[at]][0]))
            accesses.setdefault(
                address,
                {
                    "address": address,
                    "width": width,
                    "how": how,
                    "base": (names[order[at]][0], base - order[at]) if at >= 0 else None,
                },
            )
    return {
        "symbols": symbols,
        "order": order,
        "addresses": sorted(accesses),
        "accesses": accesses,
        "pairs": sorted(pairs),
    }


def _object_usage(name: str, objects: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for version, table in objects.items():
        if name not in table["symbols"]:
            continue
        start = table["symbols"][name]
        end = bisect_right(table["order"], start)
        after = table["order"][end] if end < len(table["order"]) else start + 1
        lo, hi = (bisect_right(table["addresses"], a - 1) for a in (start, after))
        result[version] = [table["accesses"][a] for a in table["addresses"][lo:hi]]
    return result


def enrich(
    snapshot: Snapshot,
    job: dict[str, Any],
    authored: dict[str, Any],
    catalogue: dict[str, Any],
    usages: dict[str, Any],
    disagreements: list[Any],
    objects: dict[str, Any],
) -> None:
    """Attach indexed type/ROM evidence inside the cached family worker."""
    globals_ = {d["name"] for d in job["declarations"] if d["kind"] == "global" and d.get("external")}
    job["conflicts"].extend(
        {"reason": f.reason, "name": f.unit, "path": f.path, "line": f.line, "declarations": list(f.missing)}
        for f in disagreements
        if f.unit in globals_
    )
    functions = {u["function"] for u in job["casts"]}
    job["holders"] = {n: snapshot.layout.members[n].holders() for n in functions if n in snapshot.layout.members}
    name = "struct Recovered_" + job["id"]
    if name in catalogue:
        job["conflicts"].append(
            {"reason": "generated record tag already has an owner", "record": name, "declaration": catalogue[name]}
        )
    job["type_evidence"] = {
        "functions": {n: authored["function"][n] for n in sorted(functions) if n in authored["function"]},
        "records": {n: authored["struct"][n] for n in job["records"] if n in authored["struct"]},
    }
    job["assembly"] = {
        v: {name: usages[v][name] for name in sorted({u["function"] for u in job["casts"]}) if name in usages[v]}
        for v in sorted(usages)
    }
    roots = {o for u in job["casts"] for o in u["roots"]}
    for decl in job["declarations"]:
        member = snapshot.layout.members.get(decl["name"]) if decl.get("external") else None
        if member and member.kind == "data" and decl["key"] in roots:
            job.setdefault("extents", []).extend(
                {"origin": decl["key"], "size": p.size, "kind": "symbol", "version": p.version}
                for p in member.placements
                if p.size > 0
            )
    for extent in list(job.get("extents", ())):
        prefix = f"param:{extent.get('function')}:"
        if extent["kind"] == "stride" and extent["base"].startswith(prefix):
            arg = "a" + extent["base"].removeprefix(prefix)
            rows = [functions.get(extent["function"], {}) for functions in usages.values()]
            if rows and all((arg, extent["size"]) in row.get("steps", ()) for row in rows):
                job["extents"].append({**extent, "kind": "assembly stride"})
    if not any(u["anchors"] for u in job["casts"]) and len(roots) == 1:
        origin = next(iter(roots))
        if origin.startswith("global:"):
            names = {d["name"] for d in job["declarations"] if d["key"] == origin and d.get("external")}
            if names:
                job["global_assembly"] = objects[next(iter(names))]
