# A unit's NON_MATCHING object may stand in for its matching one when the unit keeps its size: the same sections but
# a text that is no longer than the ROM's, filled with zeros up to it. Then every unit starts where it did, the link
# script needs no change and the ROM keeps its size. A draft that makes its unit's text or data longer leaves the
# matching object in: it is not in that ROM (report.txt names it). Inside a padded unit the code after the first
# draft moves, so a raw table of the ROM that names such a function is stale: the report counts the symbols that move.
# usage: nm_pick.py RAW MATCHING OUT OBJCOPY LD
import os, shutil, struct, subprocess, sys
def read(path):
    d = open(path, "rb").read()
    o = struct.unpack_from(">I", d, 32)[0]
    e, c, x = struct.unpack_from(">HHH", d, 46)
    rows = [struct.unpack_from(">10I", d, o + k * e) for k in range(c)]
    names = d[rows[x][4]:rows[x][4] + rows[x][5]]
    name = lambda r: names[r[0]:].split(b"\0")[0].decode()
    skip = (".pdr", ".comment", ".mdebug", ".reginfo", ".gnu.attributes")
    sizes = {name(r): r[5] for r in rows if r[1] in (1, 8) and name(r) not in skip}
    data = {name(r): d[r[4]:r[4] + r[5]] for r in rows if r[1] == 1 and name(r) not in skip}
    syms, table, undef = set(), {}, set()
    for k, r in enumerate(rows):
        if r[1] == 2:
            strs = d[rows[r[6]][4]:rows[r[6]][4] + rows[r[6]][5]]
            for j in range(r[5] // 16):
                n, v, z, i, _, sh = struct.unpack_from(">IIIBBH", d, r[4] + 16 * j)
                s = strs[n:].split(b"\0")[0].decode()
                table[(k, j)] = s
                if sh == 0 and s and j and i >> 4:
                    undef.add(s)
                if 0 < sh < 0xFF00 and i & 15 not in (3, 4) and s and not s.startswith(".L"):
                    syms.add((s, name(rows[sh]), v))
    relocs = set()
    for r in rows:
        if r[1] == 9 and name(rows[r[7]]) not in skip:
            for j in range(r[5] // 8):
                off, info = struct.unpack_from(">II", d, r[4] + 8 * j)
                relocs.add((name(rows[r[7]]), off, info & 255, table.get((r[6], info >> 8), "")))
    return sizes, syms, data, relocs, undef
raw, matching, out, objcopy, ld = sys.argv[1:6]
a, b = read(raw), read(matching)
rest = lambda sizes: {k: v for k, v in sizes.items() if k != ".text"}
missing = sorted(a[4] - b[4] - {x[0] for x in b[1]})
if missing:
    # a draft naming a symbol the unit does not (a fuzzy draft's own data name) cannot link to this ROM
    verdict, pick = "left-out undefined " + " ".join(missing[:3]), matching
elif a == b:
    verdict, pick = "same", matching
elif a[:2] == b[:2]:
    verdict, pick = "draft", raw
elif rest(a[0]) == rest(b[0]) and a[0].get(".text", 0) <= b[0].get(".text", 0):
    # a shorter text keeps the unit's size when zeros fill the rest: the next unit starts where it did
    # ld -r keeps the relocations, which objcopy --update-section drops from a section it replaces
    open(out + ".lds", "w").write("SECTIONS { .text : { *(.text) . = %d; } }\n" % b[0][".text"])
    subprocess.run([ld, "-r", "-T", out + ".lds", "-o", out, raw], check=True)
    os.remove(out + ".lds")
    moved = len({sym[0] for sym in a[1] ^ b[1]})
    verdict, pick = "draft-padded %d bytes, %d symbols move" % (b[0][".text"] - a[0][".text"], moved), None
else:
    grown = [k for k in sorted(set(a[0]) | set(b[0])) if a[0].get(k, 0) != b[0].get(k, 0)]
    why = " ".join("%s:%d->%d" % (k, b[0].get(k, 0), a[0].get(k, 0)) for k in grown)
    verdict, pick = "left-out " + (why or "symbols moved"), matching
if pick:
    shutil.copyfile(pick, out)
open(out + ".pick", "w").write("%s %s\n" % (verdict, out))
