# A data piece is the ROM bytes from its address to the next row of splat.yaml: splat drops string padding, a partial
# last word and whole tiny pieces, so each extracted data file is rewritten as its ROM slice, every label kept at its
# offset (from the ROM column of the line after it).
import os, re, sys
v = sys.argv[1]
rom = open("roms/baserom.%s.z64" % v, "rb").read()
rows = []
for line in open("versions/%s/splat.yaml" % v):
    m = re.match(r"\s+- \[0x([0-9A-Fa-f]+)(?:, (\w+), (\S+))?\]", line) or re.match(r"\s+start: 0x([0-9A-Fa-f]+)()()", line)
    if m:
        rows.append((int(m.group(1), 16), m.group(2), m.group(3)))
for (start, kind, name), (end, _, _) in zip(rows, rows[1:]):
    path = "asm/%s/data/%s.s" % (v, name)
    if kind not in ("data", "rodata") or not os.path.exists(path):
        continue
    head, labels, waiting = [], [], []
    for line in open(path):
        text = line.strip()
        m = re.match(r"(?:dlabel|glabel)\s+(\S+)", text)
        at = re.match(r"/\* ([0-9A-F]+) [0-9A-F]{8}", text)
        if m:
            waiting.append(m.group(1))
        elif text.startswith(".include"):
            head.append(text)
        elif at and waiting:
            labels += [(int(at.group(1), 16), n) for n in waiting]
            waiting = []
    labels += [(end, n) for n in waiting]
    out, here = head, start
    for offset, label in sorted(labels, key=lambda row: row[0]) + [(end, None)]:
        offset = min(max(offset, start), end)
        out += ["    .byte " + ", ".join("0x%02X" % b for b in rom[k : min(k + 16, offset)]) for k in range(here, offset, 16)]
        out += ["dlabel " + label] if label else []
        here = offset
    open(path, "w").write("\n".join(out) + "\n")
