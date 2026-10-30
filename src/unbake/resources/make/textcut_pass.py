# A function the ROM holds in several runs (other bytes of its file between them: a string, a static, a record) is one
# `.text.<function>` in the object. A fact `textpiece<n>_<function>` per later run makes the Makefile set TEXTCUT to
# `section:size:next` entries; each cuts the input section `.text.<section>` after `size` bytes and the rest becomes
# the new section `.text.<next>`, so the link script lays each run at its ROM address. An object whose relocations are
# gone (unrelocated) has no field that names a cut section; any relocation left against one is refused.
import struct, sys
path, cuts = sys.argv[1], sys.argv[2:]
raw = bytearray(open(path, "rb").read())
assert raw[:4] == b"\x7fELF" and raw[4] == 1 and raw[5] == 2, path + " is not a 32-bit big-endian ELF object"
shoff, shentsize, shnum, shstrndx = struct.unpack_from(">I", raw, 32)[0], struct.unpack_from(">H", raw, 46)[0], struct.unpack_from(">H", raw, 48)[0], struct.unpack_from(">H", raw, 50)[0]
sections = [list(struct.unpack_from(">IIIIIIIIII", raw, shoff + i * shentsize)) for i in range(shnum)]
names = bytearray(raw[sections[shstrndx][4]:sections[shstrndx][4] + sections[shstrndx][5]])
def called(section):
    return names[section[0]:names.index(0, section[0])].decode()
for cut in cuts:
    old, size, new = cut.split(":")
    size = int(size)
    index = next((i for i, section in enumerate(sections) if called(section) == ".text." + old), None)
    if index is None:
        sys.exit("%s: no section .text.%s to cut at %d" % (path, old, size))
    head = sections[index]
    if not 0 < size < head[5]:
        sys.exit("%s: .text.%s is %d bytes, not more than the cut at %d" % (path, old, head[5], size))
    if any(section[1] in (4, 9) and section[7] == index and section[5] for section in sections):
        sys.exit("%s: .text.%s still has relocations: a cut section needs an unrelocated object" % (path, old))
    while len(raw) % 16:
        raw.append(0)
    rest = bytes(raw[head[4] + size:head[4] + head[5]])
    sections.append([len(names), head[1], head[2], 0, len(raw), len(rest), 0, 0, 4, 0])
    names += (".text." + new).encode() + b"\0"
    raw += rest
    head[5] = size
    head[8] = min(head[8], 4)
    for section in sections:
        if section[1] == 2:
            for at in range(section[4], section[4] + section[5], 16):
                value, shndx = struct.unpack_from(">I", raw, at + 4)[0], struct.unpack_from(">H", raw, at + 14)[0]
                if shndx == index and value >= size:
                    struct.pack_into(">I", raw, at + 4, value - size)
                    struct.pack_into(">H", raw, at + 14, len(sections) - 1)
while len(raw) % 4:
    raw.append(0)
sections[shstrndx][4], sections[shstrndx][5] = len(raw), len(names)
raw += names
while len(raw) % 4:
    raw.append(0)
struct.pack_into(">I", raw, 32, len(raw))
struct.pack_into(">H", raw, 48, len(sections))
for section in sections:
    raw += struct.pack(">IIIIIIIIII", *section)
open(path, "wb").write(raw)
