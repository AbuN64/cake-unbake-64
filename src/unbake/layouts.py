"""`unbake types`: propose record layouts from the loads and stores the assembly makes, compare them with the C
headers, and say which open header requests the evidence answers. It only proposes: no header is written.

The evidence is accesses.py (one function at a time). A record is a named struct when a prototype, a global's
declaration or a pointer field's type names it, otherwise a global symbol or a function argument."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from unbake import accesses, cheaders, symbols
from unbake.accesses import Access, Base
from unbake.contracts import Finding, Refusal

Json = dict[str, Any]  # a mutable row of the result

_REQUEST_OFFSET = re.compile(r"\+(0x[0-9A-Fa-f]+)\b")
_SEMANTIC = re.compile(
    r"unresolved|semantic|pointee|consumer evidence|to confirm|likely|derive from|identity unresolved",
    re.IGNORECASE,
)


# ---- naming and typing bases


def _strip(name: str) -> str:
    return re.sub(r"_us_rev1$", "", name)


class Typing:
    """What the headers say a base is: the struct tag its pointer or object type names, or None."""

    def __init__(self, headers: cheaders.Headers) -> None:
        self.h = headers
        self._memo: dict[Base, str | None] = {}

    def param(self, function: str, index: int) -> cheaders.CType | None:
        proto = self.h.prototypes.get(function) or self.h.prototypes.get(_strip(function))
        if proto is None or index >= len(proto.params):
            return None
        return self.h.resolve(cheaders.parameter_type(proto.params[index]))

    def _pointee(self, ctype: cheaders.CType | None) -> str | None:
        if ctype is None or ctype.kind != "ptr" or not ctype.pointee:
            return None
        struct = self.h.struct(ctype.pointee)
        return struct.name if struct is not None else None

    def of(self, base: Base) -> str | None:
        if base in self._memo:
            return self._memo[base]
        self._memo[base] = None
        found = self._of(base)
        self._memo[base] = found
        return found

    def _of(self, base: Base) -> str | None:
        kind = base[0]
        if kind == "arg":
            return self._pointee(self.param(base[1], base[2]))
        if kind == "sym":
            text = self.h.objects.get(base[1]) or self.h.objects.get(_strip(base[1]))
            ctype = self.h.resolve(text) if text else None
            if ctype is not None and ctype.kind == "struct" and ctype.struct:
                struct = self.h.struct(ctype.struct)
                return struct.name if struct is not None else None
            return None
        if kind == "load":
            parent, offset = base[1], base[2]
            if parent[0] == "sym":
                text = self.h.objects.get(parent[1]) or self.h.objects.get(_strip(parent[1]))
                ctype = self.h.resolve(text) if text else None
                if ctype is not None and ctype.kind == "ptr" and offset == 0:
                    return self._pointee(ctype)
            owner = self.of(parent)
            struct = self.h.struct(owner) if owner else None
            if struct is None or not struct.complete:
                return None
            hit = self.h.locate(struct, offset)
            return self._pointee(hit[1].type) if hit is not None else None
        if kind == "idx":
            parent, offset = base[1], base[2]
            owner = self.of(parent)
            struct = self.h.struct(owner) if owner else None
            if struct is None or not struct.complete:
                return None
            hit = self.h.locate(struct, offset)
            if hit is not None and hit[1].type.array and hit[1].type.kind == "struct" and hit[1].type.struct:
                inner = self.h.struct(hit[1].type.struct)
                return inner.name if inner is not None else None
            if offset == 0 and parent[0] in ("arg", "load"):  # an array reached through a typed pointer
                return owner
            return None
        return None


def _describe(base: Base) -> str:
    kind = base[0]
    if kind == "sym":
        return str(base[1])
    if kind == "arg":
        return f"{base[1]}#arg{base[2]}"
    if kind == "load":
        return f"{_describe(base[1])}->@{base[2]:#x}"
    if kind == "idx":
        return f"{_describe(base[1])}[]@{base[2]:#x}"
    return "stack"


def _root(base: Base) -> Base:
    return _root(base[1]) if base[0] in ("load", "idx") else base


# ---- inference


def field_type(width: int, kinds: set[str], pointer: bool) -> tuple[str, str]:
    """(C type, note) of a field read and written with these opcode kinds."""
    floats = {k for k in kinds if k.startswith("f")}
    note = ""
    if floats and len(floats) < len(kinds):
        note = "float and integer opcodes"
    if width == 8 and floats:
        return "f64", note
    if width == 4 and floats and not pointer:
        return "f32", note
    if pointer and width == 4:
        return "void *", note
    if width == 1:
        return ("s8" if "s8" in kinds else "u8"), note
    if width == 2:
        return ("s16" if "s16" in kinds and "u16" not in kinds else "u16" if "u16" in kinds else "s16"), note
    if width == 4:
        return "s32", note
    return "s64" if width == 8 else f"u8[{width}]", note


class Record:
    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind  # struct (named by headers) | global | arg
        self.items: dict[tuple[int, int], dict[str, Any]] = {}
        self.pointers: dict[int, set[str]] = defaultdict(set)
        self.roots: set[str] = set()

    def add(self, access: Access, root: str) -> None:
        item = self.items.setdefault(
            (access.offset, access.width), {"kinds": set(), "loads": 0, "stores": 0, "functions": {}}
        )
        item["kinds"].add(access.kind)
        item["stores" if access.store else "loads"] += 1
        item["functions"].setdefault(access.function, access.pc)
        self.roots.add(root)

    def fields(self, names: Mapping[int, str]) -> tuple[list[Json], list[Json]]:
        """(fields, overlaps): one field per distinct start offset, the widest access that starts there; an access
        that starts inside a field already placed is an overlap, not a second field."""
        by_offset: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for (offset, width), item in self.items.items():
            by_offset[offset].append((width, item))
        out: list[Json] = []
        overlaps: list[Json] = []
        end = -1
        for offset in sorted(by_offset):
            options = sorted(by_offset[offset], key=lambda w: -w[0])
            width = options[0][0]
            kinds = set().union(*(i["kinds"] for _, i in options if _ == width))
            functions: dict[str, int] = {}
            for _, item in options:
                functions.update(item["functions"])
            pointer = bool(self.pointers.get(offset)) and width == 4
            ctype, note = field_type(width, kinds, pointer)
            row: Json = {
                "offset": offset,
                "width": width,
                "type": ctype,
                "name": names.get(offset, f"unk{offset:X}"),
                "pointer": pointer,
                "pointer_evidence": sorted(self.pointers.get(offset, ())),
                "opcodes": sorted(kinds),
                "loads": sum(i["loads"] for _, i in options),
                "stores": sum(i["stores"] for _, i in options),
                "functions": sorted(functions)[:6],
                "function_count": len(functions),
            }
            if len(options) > 1:
                row["other_widths"] = sorted({w for w, _ in options[1:]})
            if note:
                row["note"] = note
            if offset < end:
                row["overlaps_field_ending"] = end
                overlaps.append(row)
                continue
            out.append(row)
            end = offset + width
        return out, overlaps


def _gaps(fields: list[Json]) -> list[Json]:
    out, at = [], 0
    for f in fields:
        if f["offset"] > at:
            out.append({"offset": at, "size": f["offset"] - at})
        at = max(at, f["offset"] + f["width"])
    return out


def _proposal(name: str, fields: list[Json], size: int | None) -> str:
    lines = [f"struct {name} {{"]
    at = 0
    for f in fields:
        if f["offset"] > at:
            lines.append(f"    u8 pad{at:X}[0x{f['offset'] - at:X}];")
        ctype = f["type"]
        decl = f"{ctype}{'' if ctype.endswith('*') else ' '}{f['name']};"
        lines.append(f"    {decl}  /* +0x{f['offset']:X} */")
        at = f["offset"] + f["width"]
    if size is not None and size > at:
        lines.append(f"    u8 pad{at:X}[0x{size - at:X}];")
    lines.append("};")
    return "\n".join(lines)


# ---- comparison with headers


def compare(headers: cheaders.Headers, tag: str, fields: list[Json]) -> tuple[list[Json], Json]:
    """Every way the inferred fields disagree with the header struct `tag`, and the header's own facts."""
    struct = headers.struct(tag)
    if struct is None:
        return [], {}
    facts: Json = {"path": struct.path, "size": struct.size, "complete": struct.complete}
    issues: list[Json] = []
    if not struct.complete:
        return [{"kind": "layout-unknown", "detail": "the header struct has a field this reader cannot size"}], facts
    for f in fields:
        offset, width = f["offset"], f["width"]
        kinds = set(f["opcodes"])
        base: Json = {"offset": offset, "width": width, "access": f["type"], "functions": f["functions"]}
        if struct.size is not None and offset + width > struct.size:
            issues.append({**base, "kind": "past-end", "detail": f"struct ends at 0x{struct.size:X}"})
            continue
        hit = headers.locate(struct, offset)
        if hit is None:
            issues.append({**base, "kind": "unknown-field", "detail": "no header field holds this offset"})
            continue
        path, header, start = hit
        ctype = header.type
        known = {"header_field": path, "header_type": ctype.text, "header_offset": start}
        if header.pad and (ctype.array or ctype.kind != "ptr"):
            issues.append({**base, **known, "kind": "hidden-in-padding", "detail": f"inside reserved {path}"})
            continue
        element = ctype.stride if ctype.array and ctype.stride else ctype.size or 0
        place = start + ((offset - start) // element * element if ctype.array and element else 0)
        if offset != place:
            issues.append(
                {**base, **known, "kind": "mid-field", "detail": f"starts {offset - place} bytes into {path}"}
            )
            continue
        if ctype.kind == "struct":
            issues.append({**base, **known, "kind": "width", "detail": f"{path} is a record of {ctype.size}"})
            continue
        size = ctype.stride if ctype.array and ctype.stride else ctype.size
        if size is not None and size != width and ctype.kind != "void":
            issues.append({**base, **known, "kind": "width", "detail": f"header {ctype.text} is {size} bytes wide"})
            continue
        if f["pointer"] and ctype.kind != "ptr":
            issues.append({**base, **known, "kind": "pointer-as-int", "detail": "the loaded word is used as a pointer"})
        elif ctype.kind == "ptr" and (width != 4 or any(k.startswith("f") for k in kinds)):
            issues.append({**base, **known, "kind": "int-as-pointer", "detail": "accessed as a non-pointer scalar"})
        elif any(k.startswith("f") for k in kinds) and ctype.kind == "int":
            issues.append({**base, **known, "kind": "float-as-int", "detail": "accessed with float opcodes"})
        elif ctype.kind == "int" and width in (1, 2):
            wants = {"s8", "s16"} & kinds
            wants_u = {"u8", "u16"} & kinds
            if (wants and ctype.signed is False) or (wants_u and ctype.signed is True):
                issues.append({**base, **known, "kind": "signedness", "detail": f"opcodes {sorted(kinds)}"})
    return issues, facts


# ---- the whole run


def _function_paths(root: Path, version: str) -> list[Path]:
    return sorted((root / "asm" / version).glob("*.s"))


def collect(root: Path, version: str) -> list[accesses.Function]:
    out: list[accesses.Function] = []
    for path in _function_paths(root, version):
        out.extend(accesses.read(path))
    return out


def _spacing(table: Mapping[str, symbols.Symbol]) -> dict[str, int]:
    """name -> distance to the next data symbol's address: a bound on the record, never its size."""
    data = sorted({(v[0], n) for n, v in table.items() if v[3] != "func"})
    out: dict[str, int] = {}
    for (vram, name), following in zip(data, [*data[1:], (0, "")], strict=True):
        if following[0] > vram:
            out.setdefault(name, following[0] - vram)
    return out


def infer(
    root: Path,
    version: str,
    functions: Iterable[accesses.Function] | None = None,
    headers: cheaders.Headers | None = None,
) -> Json:
    """Every record the assembly of one version shows, with its inferred fields and, for header-named records, the
    disagreements with the header."""
    functions = list(functions) if functions is not None else collect(root, version)
    if headers is None:
        headers = cheaders.Headers()
        headers.read(root)
    typing = Typing(headers)
    table: dict[str, symbols.Symbol] = {}
    for name in (symbols.path(version), symbols.aliases_path(version)):
        file = root / name
        if file.exists():
            table.update(symbols.parse(file.read_text(errors="replace"), name))
    spacing = _spacing(table)
    records: dict[str, Record] = {}

    def record_of(base: Base) -> Record:
        tag = typing.of(base)
        if tag is not None:
            key, kind = tag, "struct"
        elif _root(base)[0] == "sym":
            key, kind = f"sym:{_describe(base)}", "global"
        else:
            key, kind = f"arg:{_describe(base)}", "arg"
        return records.setdefault(key, Record(key, kind))

    for fn in functions:
        for a in fn.accesses:
            record_of(a.base).add(a, _describe(_root(a.base)))
    for fn in functions:
        for p in fn.pointers:
            if p.why == "call":
                callee = typing.param(p.callee or "", p.arg)
                if callee is None or callee.kind != "ptr":
                    continue
            record_of(p.base).pointers[p.offset].add(p.why)

    names: dict[str, dict[int, str]] = {}
    out_records: list[Json] = []
    summary: dict[str, int] = defaultdict(int)
    for key in sorted(records):
        rec = records[key]
        struct = headers.struct(key) if rec.kind == "struct" else None
        if struct is not None:
            names[key] = {f.offset: f.name for f in struct.fields if not f.pad and f.name and f.type.kind != "struct"}
        fields, overlaps = rec.fields(names.get(key, {}))
        observed = max((f["offset"] + f["width"] for f in fields), default=0)
        symbol_name = key[4:] if key.startswith("sym:") and "->" not in key and "[]" not in key else None
        row: Json = {
            "name": key,
            "kind": rec.kind,
            "roots": sorted(rec.roots)[:6],
            "root_count": len(rec.roots),
            "size": {
                "observed": observed,
                "header": struct.size if struct is not None else None,
                "symbol_spacing": spacing.get(symbol_name) if symbol_name else None,
            },
            "fields": fields,
            "overlaps": overlaps,
            "gaps": _gaps(fields),
            "proposal": _proposal(key.split(":", 1)[-1] if rec.kind != "struct" else key, fields, None),
        }
        if struct is not None:
            issues, facts = compare(headers, key, fields)
            row["header"] = facts
            row["diff"] = sorted(issues, key=lambda i: (i.get("offset", -1), i["kind"]))
            summary["records_with_diff"] += bool(issues)
        summary[rec.kind] += 1
        out_records.append(row)
    return {
        "version": version,
        "summary": {
            "functions": len(functions),
            "accesses": sum(len(f.accesses) for f in functions),
            "records": len(out_records),
            "struct_records": summary["struct"],
            "global_records": summary["global"],
            "arg_records": summary["arg"],
            "records_with_diff": summary["records_with_diff"],
        },
        "records": out_records,
    }


# ---- requests


_FILLER = frozenset({"offset", "offsets", "field", "word", "words", "bytes", "byte", "and", "the", "type", "size"})
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_OFFSET = re.compile(r"(?:\+|\boffsets? ?=?|\bat |(?<=[A-Za-z0-9_]) )0x([0-9A-Fa-f]{1,4})\b")


def parse_requests(text: str) -> list[Json]:
    """One row per request bullet: its line, settled flag, the identifiers it mentions in order, the offsets it asks
    about (`+0x..`, `offset 0x..`, or a bare 0x.. right after a name) and the functions it cites."""
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("- "):
            continue
        settled = line.startswith("- SETTLED")
        body = line[2:]
        if settled:  # the "(...)" after SETTLED may hold names of its own: look only past its closing "):"
            close = body.find("): ")
            body = body[close + 3 :] if close >= 0 else body
        offsets = sorted({int(m, 16) for m in _OFFSET.findall(body)})
        anchors: list[str] = []  # the names right before an offset: what the line says the offset belongs to
        for found in _OFFSET.finditer(body):
            window = body[max(0, found.start() - 48) : found.start()]
            names = [w for w in _WORD.findall(window) if w.lower() not in _FILLER]
            anchors += reversed(names[-3:])
        words = list(dict.fromkeys(anchors))
        rows.append(
            {
                "line": number,
                "settled": settled,
                "names": words,
                "offsets": offsets,
                "functions": sorted(set(re.findall(r"\b(func_[0-9A-Fa-f]{8}\w*)", body))),
                "semantic": bool(_SEMANTIC.search(body)),
            }
        )
    return rows


def _fits(field: Json, requested: int) -> bool:
    return bool(field["offset"] <= requested < field["offset"] + field["width"])


def _cover(record: Json, offsets: list[int]) -> tuple[list[Json], list[str]]:
    hits, missing = [], []
    for offset in offsets:
        field = next((f for f in record["fields"] if _fits(f, offset)), None) or next(
            (f for f in record["overlaps"] if _fits(f, offset)), None
        )
        if field is None:
            missing.append(f"no access at +0x{offset:X}")
        else:
            hits.append(
                {
                    "offset": f"+0x{offset:X}",
                    "type": field["type"],
                    "width": field["width"],
                    "pointer": field["pointer"],
                    "functions": field["functions"][:3],
                    "at_field_start": field["offset"] == offset,
                }
            )
    return hits, missing


def settle(requests: list[Json], inferred: Json, headers: cheaders.Headers) -> Json:
    """For every open request: the record it names, and whether the inferred fields answer it. A record is found by a
    struct or global name in the line; failing that, by the argument records of the functions the line cites (which
    is weaker: nothing proves the argument is the record the line means)."""
    records = {r["name"]: r for r in inferred["records"]}
    for r in inferred["records"]:  # a global is named by its symbol alone
        if r["name"].startswith("sym:"):
            records.setdefault(r["name"][4:], r)
    by_function: dict[str, list[Json]] = defaultdict(list)
    for r in inferred["records"]:
        if r["kind"] == "arg":
            by_function[r["name"][4:].split("#")[0]].append(r)
    rows: list[Json] = []
    for req in requests:
        if req["settled"]:
            continue
        row: Json = {"line": req["line"], "offsets": [f"+0x{o:X}" for o in req["offsets"]]}
        if not req["offsets"]:
            row.update(status="no-offset", missing=["the request names no offset"])
            rows.append(row)
            continue
        candidates: list[tuple[Json, str]] = []
        for name in req["names"]:
            tag = headers._tag(name) if name in headers._raw or name in headers.typedefs else name
            found = records.get(tag) or records.get(name)
            if found is not None and all(found is not c[0] for c in candidates):
                candidates.append((found, "named"))
        best: tuple[Json, str] | None = None
        best_hits: list[Json] = []
        best_missing: list[str] = []
        for record, via in candidates:
            hits, missing = _cover(record, req["offsets"])
            if best is None or len(hits) > len(best_hits):
                best, best_hits, best_missing = (record, via), hits, missing
        if best is None or not best_hits:
            for name in req["functions"]:
                for record in [*by_function.get(name, ()), *by_function.get(_strip(name), ())]:
                    hits, missing = _cover(record, req["offsets"])
                    if len(hits) > len(best_hits):
                        best, best_hits, best_missing = (record, "function-argument"), hits, missing
        if best is None:
            row.update(status="no-record", missing=["no named record or cited function has a matching access"])
            rows.append(row)
            continue
        record, via = best
        row.update(record=record["name"], via=via, evidence=best_hits)
        named = [f for h in best_hits for f in h["functions"] if f in req["functions"]]
        row["names_a_requested_function"] = bool(named)
        if best_missing:
            row.update(status="partial" if best_hits else "no-evidence", missing=best_missing)
        else:
            semantic = req["semantic"]
            row.update(status="layout" if semantic else "answered", missing=["semantic identity"] if semantic else [])
            if via != "named":
                row["status"] = "layout" if semantic else "answered-via-function"
        diffs = [d for d in record.get("diff", ()) if f"+0x{d.get('offset', -1):X}" in row["offsets"]]
        if diffs:
            row["header_disagrees"] = [{k: d[k] for k in ("offset", "kind", "detail")} for d in diffs]
        rows.append(row)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    return {
        "open": len(rows),
        "settled_skipped": sum(1 for r in requests if r["settled"]),
        "answered": counts["answered"],
        "answered_via_function": counts["answered-via-function"],
        "layout_only": counts["layout"],
        "partial": counts["partial"],
        "no_evidence": counts["no-evidence"],
        "no_record": counts["no-record"],
        "no_offset": counts["no-offset"],
        "rows": rows,
    }


# ---- command


def text_view(inferred: Json, only: set[str] | None = None) -> str:
    lines = []
    for r in inferred["records"]:
        if r["kind"] == "arg" or (only and r["name"] not in only):
            continue
        size = r["size"]
        lines.append(
            f"{r['name']} [{r['kind']}] observed 0x{size['observed']:X}"
            + (f" header 0x{size['header']:X}" if size["header"] is not None else "")
            + (f" symbol-spacing 0x{size['symbol_spacing']:X}" if size["symbol_spacing"] else "")
        )
        lines.append(r["proposal"])
        for d in r.get("diff", ()):
            lines.append(f"  ! +0x{d.get('offset', 0):X} {d['kind']}: {d['detail']}")
        lines.append("")
    return "\n".join(lines)


def types_command(params: Json) -> Json:
    root = Path(params["project"])
    version = params.get("version") or "us-rev1"
    if not (root / "asm" / version).is_dir():
        raise Refusal(Finding("types.input", f"no extracted assembly at asm/{version}: run `unbake setup` first"))
    headers = cheaders.Headers()
    headers.read(root)
    inferred = infer(root, version, None, headers)
    result: Json = {"version": version, "summary": inferred["summary"]}
    wanted = set(params.get("record") or ())
    records = [r for r in inferred["records"] if not wanted or r["name"] in wanted]
    if not params.get("all"):
        records = [r for r in records if r["kind"] != "arg" or wanted]
    result["records"] = records
    if params.get("requests"):
        try:
            text = Path(params["requests"]).read_text(errors="replace")
        except OSError as error:
            raise Refusal(Finding("types.input", str(error))) from error
        result["requests"] = settle(parse_requests(text), inferred, headers)
    if params.get("text"):
        Path(params["text"]).write_text(text_view(inferred, wanted or None))
    if params.get("out"):
        Path(params["out"]).write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        result = {k: v for k, v in result.items() if k != "records"} | {"written": str(params["out"])}
        if "requests" in result:
            result["requests"] = {k: v for k, v in result["requests"].items() if k != "rows"}
    return result
