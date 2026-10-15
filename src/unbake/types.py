"""Persist type evidence and propose declarations from assembly."""

import json
import re
import tomllib
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from unbake import config, effort, headers, infer, store
from unbake.contracts import Finding, Json, Refusal, Snapshot, code_identity, digest
from unbake.families import layoutattr
from unbake.symbols import by_address

if TYPE_CHECKING:
    from pycparser import c_ast

Spelled = Mapping[str, list[tuple[str, int, str]]]  # headers.externs: name -> (path, line, spelling) of its externs
RecordIndex = tuple[dict[str, "c_ast.Node"], dict[str, "c_ast.Node"], set[str]]  # typedefs, records, duplicate tags


def anonymous_record(node: "c_ast.Node") -> "c_ast.Node | None":
    """The unnamed struct or union a typedef's declarator chain ends in, else None."""
    from pycparser import c_ast

    while isinstance(node, c_ast.TypeDecl):
        node = node.type
    return node if isinstance(node, (c_ast.Struct, c_ast.Union)) and node.name is None else None


def load(snapshot: Snapshot) -> Json:
    with effort.stage("types.load"):
        raw = snapshot.peek("types.toml")
        if raw is None:
            raise Refusal(
                Finding("config.missing", "types.toml does not exist", path="types.toml", action="unbake setup")
            )
        cached = store.content(snapshot.config).cached
        return effort.memo(("types", digest(raw)), lambda: config.toml("types", raw, "types.toml", cache=cached))


def authority(snapshot: Snapshot) -> Json:
    """The type map with the signatures of the landed sources laid over it: what a header judged mid-drain meets."""

    def produce() -> Json:
        known = load(snapshot)
        landed = {n: {"signature": t, "evidence": "landed"} for n, t in headers.landed(snapshot).items()}
        return {**known, "function": {**known["function"], **landed}}

    pins = (headers.pins(snapshot, ["types.toml"]), headers.landed_sources(snapshot)[1])
    return effort.memo(("authority", str(snapshot.config.project.root), pins), produce)


def _dump(
    doc: Json,
) -> bytes:  # inline tables written directly: the document is megabytes and a TOML library walks it for seconds
    lines = [f"schema = {doc['schema']}"]
    for kind in ("function", "global", "struct"):
        lines.append(f"\n[{kind}]")
        lines.extend(
            f"{json.dumps(name)} = {{ "
            + ", ".join(f"{key} = {json.dumps(value)}" for key, value in sorted(row.items()))
            + " }"
            for name, row in sorted(doc[kind].items())
        )
    return ("\n".join(lines) + "\n").encode()


_KEYWORDS = frozenset(
    (
        "extern",
        "const",
        "volatile",
        "unsigned",
        "signed",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "void",
        "struct",
        "union",
        "enum",
    )
)


def scan(snapshot: Snapshot) -> bytes:
    """Setup's type map: globals from data placements, header prototypes and types (authored) and landed definitions."""
    with effort.stage("types.scan"):
        try:
            current = snapshot.read("types.toml")
        except FileNotFoundError:
            load(snapshot)  # refuses by name
            raise
        key = digest(
            (
                snapshot.layout.digest,
                snapshot.config.project.names_from,
                current,
                code_identity(_scan),
                [(p, digest(snapshot.read(p))) for p in headers.sources(snapshot)],
                [(p, digest(snapshot.read(p))) for p in headers.sources(snapshot, "src", ".c")],
            )
        )
        return store.cached(snapshot.config, "types-scan", key, lambda: _scan(snapshot))


def _scan(snapshot: Snapshot) -> bytes:
    stale = snapshot.read("types.toml")  # read without validation: this scan replaces a file the schema refuses
    schema = int(stale.split(b"=", 1)[1].split(b"\n", 1)[0])
    doc: dict[str, Any] = {"schema": schema, "struct": {}, "global": {}, "function": {}}
    for name, text in headers.landed(snapshot).items():
        doc["function"][name] = {"signature": text, "evidence": "landed"}
    for name, (_, _, text) in headers.catalog(snapshot).items():
        if re.match(r"(?:typedef|struct|union|enum)\b", text):
            doc["struct"].setdefault(name.rpartition(" ")[2], {"declaration": text, "evidence": "authored"})
        elif re.search(rf"\b{re.escape(name)}\s*\(", text):
            signature = re.sub(r"^extern\s+|\s*;$", "", text)
            doc["function"].setdefault(name, {"signature": signature, "evidence": "authored"})
        else:
            doc["global"][name] = {"declaration": text, "evidence": "authored"}
    for kind, rows in tomllib.loads(stale.decode()).items():  # a row nothing in the C produces is authored: it stays
        if kind in ("struct", "global", "function"):
            for name, row in rows.items():
                if row.get("evidence") not in ("landed", "declared"):  # those two are recomputed above and below
                    doc[kind].setdefault(name, row)
    known = _KEYWORDS | doc["struct"].keys()
    agreed = {
        n: text
        for n, text in headers.disagreements(
            snapshot, {n for n, r in doc["function"].items() if r["evidence"] == "landed"}
        )[0].items()
        if not set(re.findall(r"[A-Za-z_]\w*", text.replace("@", ""))) - known
    }
    for name, member in sorted(snapshot.layout.members.items()):
        if member.kind == "function" and name in agreed and name not in doc["function"]:
            doc["function"][name] = {"signature": agreed[name].replace("@", name, 1), "evidence": "declared"}
    return _dump(doc)


def conflicts(snapshot: Snapshot, paths: Collection[str] | None = None) -> list[Finding]:
    """The names the landed C declares with different types and defines nowhere: a finding per declaration."""
    with effort.stage("types.conflicts"):
        defined = {n for n, r in load(snapshot)["function"].items() if r["evidence"] == "landed"}
        return headers.disagreements(snapshot, defined, *(() if paths is None else (paths,)))[1]


def _tables(snapshot: Snapshot, version: str) -> tuple[dict[str, Json], dict[int, list[Any]]]:
    """(uses, readers) of one version, cached: the usage of each function, and per address who touches it."""

    def produce() -> tuple[dict[str, Json], dict[int, list[Any]]]:
        uses: dict[str, Json] = {row[0]: row[4] for row in infer.scan(snapshot)[version]}
        readers: defaultdict[int, list[Any]] = defaultdict(list)
        for name, use in uses.items():
            for address, width, _, how, base in use["accesses"]:
                readers[address].append((name, how, width, base))
        return uses, dict(readers)

    key = digest((snapshot.layout.digest, snapshot.versions[version].rom_sha256, version, code_identity(_tables)))
    return store.cached_object(snapshot.config, "usage", key, produce)


def _spans(use: Json) -> list[list[int]]:
    """[start, end, stores, fills] runs of one function's stores and fills, merged across gaps of at most 0x200."""
    taken = {a for a, _, _, how, _ in use["accesses"] if how == "taken"}
    points = [(a, a + w, 1, 0) for a, w, m, how, _ in use["accesses"] if m[0] == "s" and how != "taken"]
    for _, args in use["calls"]:
        given = dict(args)
        if given.get(0) in taken and given.get(2):
            points.append((given[0], given[0] + given[2], 0, 1))
    spans: list[list[int]] = []
    for start, end, stores, fills in sorted(points):
        if spans and start - spans[-1][1] <= 0x200:
            spans[-1] = [spans[-1][0], max(spans[-1][1], end), spans[-1][2] + stores, spans[-1][3] + fills]
        else:
            spans.append([start, end, stores, fills])
    return [s for s in spans if s[2] >= 4 or s[3]]


def _landed(snapshot: Snapshot, name: str) -> bool:
    kind = config.kinds().get(snapshot.layout.members[name].state)
    return bool(kind and kind["decompiled"])


def _rows(
    snapshot: Snapshot, version: str, items: list[tuple[Any, ...]], spelled: Spelled | None = None
) -> list[dict[str, Any]]:
    """One row per (address, how, width, base): its names, readers (landed first), spellings, span, base and stride."""
    uses, readers = _tables(snapshot, version)
    symbols = snapshot.versions[version].symbols
    spelled = headers.externs(snapshot) if spelled is None else spelled
    names = by_address(symbols)
    order = sorted(names)
    rows: list[dict[str, Any]] = []
    for address, how, width, base in items:
        users = sorted({n for n, *_ in readers.get(address, ())}, key=lambda n: (not _landed(snapshot, n), n))
        span = next(([a, b - a, n, s, f] for n in users for a, b, s, f in _spans(uses[n]) if a <= address < b), None)
        at = bisect_right(order, base) - 1 if base is not None else -1
        stride = next((c for n in users for k, c in uses[n]["steps"] if k == base), None) if base else None
        rows.append(
            {
                "address": address,
                "names": sorted(names[address]),
                "how": how,
                "width": width,
                "readers": [(n, _landed(snapshot, n)) for n in users[:5]],
                "landed_readers": sum(_landed(snapshot, n) for n in users),
                "spellings": [s for n in names[address] for s in spelled.get(n, ())][:3],
                "span": dict(zip(("start", "size", "writer", "stores", "fills"), span, strict=True)) if span else None,
                "base": (names[order[at]][0], base - order[at]) if at >= 0 else None,
                "stride": stride,
            }
        )
    return rows


def usage(snapshot: Snapshot, member: str, version: str) -> Json:
    """What one function does with memory: per address it touches the row of _rows, its argument accesses and calls."""
    with effort.stage("types.usage"):
        uses, _ = _tables(snapshot, version)
        if member not in uses:
            raise Refusal(Finding("types.usage", f"{member} is no function of {version}", unit=member))
        use = uses[member]
        items = sorted({(a, how, w, base) for a, w, _, how, base in use["accesses"]}, key=lambda r: r[:3])
        callers = [n for n, u in uses.items() if any(c == member for c, _ in u["calls"]) and _landed(snapshot, n)]
        args = {
            f"a{n}": {
                "offsets": sorted({o for k, o, _ in use["args"] if k == n}),
                "stride": next((c for k, c in use["steps"] if k == f"a{n}"), None),
                "landed_callers": callers[:5],
            }
            for n in sorted({k for k, _, _ in use["args"]})
        }
        return {
            "addresses": _rows(snapshot, version, items),
            "args": args,
            "calls": [(c, {f"a{n}": v for n, v in a}) for c, a in use["calls"]],
        }


def usage_at(snapshot: Snapshot, name: str, spellings: Spelled | None = None) -> Json:
    """Per version that names the symbol, the rows of _rows for every address inside the symbol's extent."""
    with effort.stage("types.usage_at"):
        found: dict[str, list[dict[str, Any]]] = {}
        for version, record in snapshot.versions.items():
            if name in record.symbols:
                start = record.symbols[name]
                after = min((a for a in record.symbols.values() if a > start), default=start + 1)
                _, readers = _tables(snapshot, version)
                found[version] = _rows(
                    snapshot,
                    version,
                    [(a, *readers[a][0][1:]) for a in sorted(readers) if start <= a < after],
                    spellings,
                )
        return found


def _record_index(ast: "c_ast.FileAST") -> RecordIndex:
    """Collect the declaration context independently of requested record layouts."""
    from pycparser import c_ast

    typedefs = {n.name: n.type for n in ast.ext if isinstance(n, c_ast.Typedef)}
    records: dict[str, c_ast.Node] = {}
    duplicates: set[str] = set()

    class Collect(c_ast.NodeVisitor):
        def visit_Struct(self, node: c_ast.Struct) -> None:
            if node.name and node.decls is not None:
                if "struct " + node.name in records and records["struct " + node.name] is not node:
                    duplicates.add("struct " + node.name)
                records["struct " + node.name] = node
            self.generic_visit(node)

        def visit_Union(self, node: c_ast.Union) -> None:
            if node.name and node.decls is not None:
                if "union " + node.name in records and records["union " + node.name] is not node:
                    duplicates.add("union " + node.name)
                records["union " + node.name] = node
            self.generic_visit(node)

    Collect().visit(ast)
    return typedefs, records, duplicates


def record_layouts(
    ast: "c_ast.FileAST",
    names: Collection[str] | None = None,
    tainted: Collection[str] = frozenset(),
    *,
    index: RecordIndex | None = None,
) -> dict[str, dict[str, Any]]:
    """Exact o32 C layouts from the header AST; unsupported layouts retain their cause.

    N64 pointers/longs are 32 bits. `aligned(N)` markers raise record or field alignment. Bitfields and
    other marked-unsupported records, `tainted` typedefs, variable dimensions and recursive by-value records
    are not inferred.
    Typedefs are expanded by the existing header canonicalizer, so signedness and pointed-to types remain significant.
    """
    from pycparser import c_ast, c_generator

    typedefs, records, duplicates = _record_index(ast) if index is None else index
    records = dict(records)
    result: dict[str, Any] = {}
    busy: set[str] = set()
    generator = c_generator.CGenerator()

    for name, node in typedefs.items():
        if (base := anonymous_record(node)) is not None:
            records.setdefault(name, base)
    primitive = {
        "char": (1, 1),
        "signed char": (1, 1),
        "unsigned char": (1, 1),
        "short": (2, 2),
        "short int": (2, 2),
        "signed short": (2, 2),
        "unsigned short": (2, 2),
        "unsigned short int": (2, 2),
        "int": (4, 4),
        "signed": (4, 4),
        "signed int": (4, 4),
        "unsigned": (4, 4),
        "unsigned int": (4, 4),
        "long": (4, 4),
        "long int": (4, 4),
        "unsigned long": (4, 4),
        "unsigned long int": (4, 4),
        "float": (4, 4),
        "double": (8, 8),
        "long long": (8, 8),
        "long long int": (8, 8),
        "unsigned long long": (8, 8),
        "unsigned long long int": (8, 8),
    }

    def number(node: c_ast.Node | None) -> int:
        if isinstance(node, c_ast.Constant) and node.type == "int":
            value = re.sub("[uUlL]+$", "", node.value)
            return int(value, 8 if re.fullmatch(r"0[0-7]+", value) else 0)
        if isinstance(node, c_ast.UnaryOp):
            if node.op == "sizeof" and isinstance(node.expr, c_ast.Typename):
                return shape(node.expr.type)[0]
            if node.op in ("+", "-"):
                return number(node.expr) * (-1 if node.op == "-" else 1)
        if isinstance(node, c_ast.BinaryOp) and node.op in ("+", "-", "*", "/", "%", "<<", ">>"):
            a, b = number(node.left), number(node.right)
            return {
                "+": lambda: a + b,
                "-": lambda: a - b,
                "*": lambda: a * b,
                "/": lambda: (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1),
                "%": lambda: a - ((abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)) * b,
                "<<": lambda: a << b,
                ">>": lambda: a >> b,
            }[node.op]()
        raise ValueError("nonconstant array extent: " + generator.visit(node) if node else "incomplete array")

    def shape(node: c_ast.Node, seen: frozenset[str] = frozenset()) -> tuple[int, int, list[dict[str, Any]]]:
        if isinstance(node, c_ast.PtrDecl):
            return 4, 4, []
        if isinstance(node, c_ast.TypeDecl):
            return shape(node.type, seen)
        if isinstance(node, c_ast.ArrayDecl):
            size, align, _ = shape(node.type, seen)
            count = number(node.dim)
            if count < 0:
                raise ValueError("negative array extent")
            return size * count, align, []
        if isinstance(node, c_ast.Enum):
            return 4, 4, []
        if isinstance(node, c_ast.IdentifierType):
            name = " ".join(node.names)
            if name in tainted:
                raise ValueError("unsupported layout: type " + name + " carries a layout attribute")
            if name in primitive:
                return (*primitive[name], [])
            if name in typedefs and name not in seen:
                return shape(typedefs[name], seen | {name})
            raise ValueError("unknown storage type: " + name)
        if isinstance(node, (c_ast.Struct, c_ast.Union)):
            name = ("struct " if isinstance(node, c_ast.Struct) else "union ") + (node.name or "")
            if node.decls is None:
                if name not in records:
                    raise ValueError("incomplete record: " + name)
                return layout(name)
            offset, alignment, wanted, cap = 0, 1, 1, 0
            fields: list[dict[str, Any]] = []
            for decl in node.decls:
                if (decl.name or "").startswith(layoutattr.MARKER):
                    what, _, value = decl.name[len(layoutattr.MARKER) :].rpartition("_")
                    if what == "unsupported":
                        raise ValueError("unsupported layout: " + value)
                    if what == "pack":
                        cap = int(value)
                    else:
                        alignment, wanted = (
                            (max(alignment, int(value)), wanted) if what == "aligned" else (alignment, int(value))
                        )
                    continue
                if decl.bitsize is not None or (
                    not decl.name and not isinstance(decl.type, (c_ast.Struct, c_ast.Union))
                ):
                    raise ValueError("bitfield or anonymous member: " + name)
                if not decl.name:  # anonymous record: its fields sit at the parent offsets; union arms overlap
                    size, align, sub = shape(decl.type, seen)
                    align, wanted = max(min(align, cap) if cap else align, wanted), 1
                    alignment = max(alignment, align)
                    at = 0 if isinstance(node, c_ast.Union) else (offset + align - 1) // align * align
                    arm = isinstance(decl.type, c_ast.Union)
                    fields.extend({**f, "offset": f["offset"] + at, **({"arm": True} if arm else {})} for f in sub)
                    offset = max(offset, size) if isinstance(node, c_ast.Union) else at + size
                    continue
                size, align, sub = shape(decl.type, seen)
                align, wanted = max(min(align, cap) if cap else align, wanted), 1
                alignment = max(alignment, align)
                at = 0 if isinstance(node, c_ast.Union) else (offset + align - 1) // align * align
                spelling = headers.canonical(decl, typedefs)
                if isinstance(decl.type, c_ast.ArrayDecl):
                    spelling = re.sub(r"\[[^]]*\]", "[" + str(number(decl.type.dim)) + "]", spelling, count=1)
                fields.append(
                    {
                        "name": decl.name,
                        "offset": at,
                        "width": size,
                        "alignment": align,
                        "type": spelling,
                        "declaration": generator.visit(decl) + ";",
                        "fields": sub,
                    }
                )
                offset = max(offset, size) if isinstance(node, c_ast.Union) else at + size
            return (offset + alignment - 1) // alignment * alignment, alignment, fields
        raise ValueError("unsupported storage: " + type(node).__name__)

    def layout(name: str) -> tuple[int, int, list[dict[str, Any]]]:
        if name in duplicates:
            raise ValueError("multiple scoped record definitions: " + name)
        if name in busy:
            raise ValueError("recursive by-value record: " + name)
        if name in result:
            row = result[name]
            if "conflict" in row:
                raise ValueError(row["conflict"])
            return row["size"], row["alignment"], row["fields"]
        busy.add(name)
        try:
            size, alignment, fields = shape(records[name])
            result[name] = {"size": size, "alignment": alignment, "fields": fields}
            return size, alignment, fields
        finally:
            busy.remove(name)

    selected = set(records) if names is None else set(names)
    for name in selected:
        if name not in records:
            continue
        try:
            layout(name)
        except (ValueError, ZeroDivisionError) as error:
            result[name] = {"conflict": str(error)}
    for name, node in typedefs.items():
        if names is not None and name not in selected:
            continue
        base = node
        while isinstance(base, c_ast.TypeDecl):
            base = base.type
        if isinstance(base, (c_ast.Struct, c_ast.Union)) and base.name:
            tag = ("struct " if isinstance(base, c_ast.Struct) else "union ") + base.name
            if tag in result:
                result[name] = {**result[tag], "canonical": tag}
    return result
