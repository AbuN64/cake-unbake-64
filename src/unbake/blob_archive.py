"""Offset/size menu archive records and section-relative alignment, independent of ROM placement."""

import struct
from typing import Any

from unbake.contracts import Json

SECTIONS = ("image", "font", "font-palette", "glyph", "language", "string", "page", "page-data", "pixel", "font-data")


def header_at(data: bytes, at: int) -> tuple[int, ...] | None:
    if not 0 <= at <= len(data) - 84:
        return None
    w = struct.unpack_from(">21I", data, at)
    if w[0] != 84 or w[20] != 0xFFFFFFFF or not all(w[i] + w[i + 1] == w[i + 2] for i in range(0, 18, 2)):
        return None
    return w


# Page widget node size by type (low half of word 3): a 44-byte header (parent, next, child, id << 16 | type, flags,
# position, extent, 0, scale x, scale y, 0) plus a fixed per-type tail.
NODE = {1: 44, 2: 76, 3: 48, 4: 60, 6: 60, 8: 56, 9: 68}


def page_nodes(data: bytes, at: int, off: int, size: int) -> dict[int, int]:
    """Walk one page: its nodes tile [off, off + size) by type size, the first is the type-2 root, and every
    parent/next/child link is 0 or another node of the same page."""
    nodes: dict[int, int] = {}
    o = off
    while o < off + size:
        kind = struct.unpack_from(">I", data, at + o + 12)[0] & 0xFFFF
        if kind not in NODE or (o == off) != (kind == 2):
            raise ValueError(("page node type", hex(off), hex(o), kind))
        nodes[o] = kind
        o += NODE[kind]
    if o != off + size:
        raise ValueError(("page not tiled by its nodes", hex(off)))
    for o in nodes:
        for link in struct.unpack_from(">3I", data, at + o):
            if link and link not in nodes:
                raise ValueError(("page link outside its nodes", hex(off), hex(o), hex(link)))
    return nodes


def archive(data: bytes, base: int) -> Json:
    h = header_at(data, base)
    if h is None:
        raise ValueError("no menu archive header at base")
    sec = {name: (base + h[2 * i], base + h[2 * i] + h[2 * i + 1]) for i, name in enumerate(SECTIONS)}
    end = sec["font-data"][1]
    if end > len(data):
        raise ValueError("archive exceeds enclosing bytes")
    if h[1] % 28 or h[3] % 32 or h[5] % 4 or h[9] % 8 or h[13] % 12:
        raise ValueError("partial descriptor record")
    recs: list[Any] = []  # (name, kind, spans, facts)

    def inside(s: str, off: int, size: int) -> tuple[int, int]:
        a, b = sec[s]
        if off < 0 or size <= 0 or a + off + size > b:
            raise ValueError((s, off, size))
        return (a + off, a + off + size)

    recs.append(("header", "header", [(base, base + 84)], {}))
    # images
    pix = []
    for i in range((sec["image"][1] - sec["image"][0]) // 28):
        s = sec["image"][0] + 28 * i
        pre = data[s : s + 8]
        depth, width, height, colors, u0, u1, pixels, palette = struct.unpack_from(">6H2I", data, s + 8)
        if pre != bytes(8) or u0 or u1 or depth not in (4, 8, 16) or not width or not height:
            raise ValueError(("image", i, pre.hex(), depth, width, height, u0, u1))
        align = {4: 16, 8: 8, 16: 4}[depth]
        stride = (width + align - 1) // align * align
        spans = [(s, s + 28), inside("pixel", pixels, stride * height * depth // 8)]
        if colors:
            spans.append(inside("pixel", palette, colors * 2))
        elif depth != 16:
            raise ValueError(("image", i, "paletted depth without colors"))
        pix += spans[1:]
        recs.append(
            (
                f"image.{i}",
                "image",
                spans,
                {"depth": depth, "width": width, "height": height, "colors": colors, "stride": stride},
            )
        )
    # fonts: descriptor, glyph table (96 x 8), bitmap, style palettes
    pal_off = struct.unpack_from(
        f">{(sec['font-palette'][1] - sec['font-palette'][0]) // 4}I", data, sec["font-palette"][0]
    )
    pi = 0
    for i in range((sec["font"][1] - sec["font"][0]) // 32):
        s = sec["font"][0] + 32 * i
        z0, z1, colors, f10, f12, styles, f16, f18, glyphs, bitmap, bsize = struct.unpack_from(">2I6H3I", data, s)
        if z0 or z1 or not colors or not styles or pi + styles > len(pal_off):
            raise ValueError(("font", i))
        stride = (colors * 2 + 7) & ~7
        # 96 glyph records (zero, bitmap offset, width, height, metrics): CI4 rows of 8 bytes (16 px) that tile
        # the font's bitmap exactly
        at = bitmap
        for g in range(96):
            zero, offset, width, height, _metric = struct.unpack_from(">HHBBH", data, sec["glyph"][0] + glyphs + 8 * g)
            if zero or offset != at or width > 16:
                raise ValueError(("glyph", i, g))
            at += 8 * height
        if at != bitmap + bsize:
            raise ValueError(("font bitmap not tiled by its glyphs", i))
        spans = [
            (s, s + 32),
            inside("glyph", glyphs, 96 * 8),
            inside("font-data", bitmap, bsize),
            (sec["font-palette"][0] + 4 * pi, sec["font-palette"][0] + 4 * (pi + styles)),
        ]
        spans += [inside("font-data", o, stride) for o in pal_off[pi : pi + styles]]
        pi += styles
        recs.append(
            (
                f"font.{i}",
                "font",
                spans,
                {"colors": colors, "styles": styles, "glyphs": 96, "metrics": [f10, f12, f16, f18]},
            )
        )
    if pi != len(pal_off):
        raise ValueError("font palette offsets left unowned")
    # text: language table + strings
    langs = struct.unpack_from(f">{(sec['language'][1] - sec['language'][0]) // 4}I", data, sec["language"][0])
    strings = data[sec["string"][0] : sec["string"][1]]
    if strings[-1:] != b"\0" or sum(langs[1::2]) != strings.count(b"\0"):
        raise ValueError("string table counts disagree with the NUL-terminated strings")
    recs.append(
        (
            "text",
            "text",
            [sec["language"], sec["string"]],
            {"languages": len(langs) // 2, "strings": strings.count(b"\0")},
        )
    )
    # pages
    nxt = 0
    for i in range((sec["page"][1] - sec["page"][0]) // 12):
        s = sec["page"][0] + 12 * i
        off, size, res = struct.unpack_from(">3I", data, s)
        if res or off != nxt:
            raise ValueError(("page", i, off, nxt, res))
        nxt = off + size
        nodes = page_nodes(data, sec["page-data"][0], off, size)
        recs.append(
            (
                f"page.{i}",
                "page",
                [(s, s + 12), inside("page-data", off, size)],
                {"nodes": len(nodes), "types": sorted(set(nodes.values()))},
            )
        )
    if base + h[14] + nxt != sec["page-data"][1]:
        raise ValueError("page data not tiled")
    # tiling
    spans = sorted(sp for r in recs for sp in r[2])
    fills, at = [], base
    for a, b in spans:
        if a < at:
            raise ValueError(("overlap", hex(a), hex(at)))
        if a > at:
            gap = data[at:a]
            origins = [low for low, high in sec.values() if low <= at < a <= high]
            boundary = any(high == a for _, high in sec.values())
            if len(origins) != 1 or (not boundary and (a - origins[0]) % 8) or a - at > 7 or any(gap):
                raise ValueError(("gap", hex(at), hex(a), gap.hex()))
            fills.append((at, a, gap[0]))
        at = b
    if at != end:
        raise ValueError(("tail", hex(at), hex(end)))
    return {"base": base, "end": end, "records": recs, "fills": fills}
