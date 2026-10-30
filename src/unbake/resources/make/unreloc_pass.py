# A unit whose file row says `unrelocated` is a relocatable object the ROM holds before its link: the relocated fields of
# its code and data stay zero (the ROM keeps the patches elsewhere) and the object carries no relocations, so the link
# lays its bytes down as they are. This zeroes the field each REL entry names (a 26-bit jump target, a 16-bit half of an
# address, a 32-bit word) and empties the REL sections. The compiler's own read-only data (a switch's jump table, a
# literal pool) is the ROM's already, in the data the file stands over, so those sections stop being loaded: the link
# places none of them.
import struct, sys
path = sys.argv[1]
raw = bytearray(open(path, "rb").read())
assert raw[:4] == b"\x7fELF" and raw[4] == 1 and raw[5] == 2, path + " is not a 32-bit big-endian ELF object"
shoff, shentsize, shnum, shstrndx = struct.unpack_from(">I", raw, 32)[0], struct.unpack_from(">H", raw, 46)[0], struct.unpack_from(">H", raw, 48)[0], struct.unpack_from(">H", raw, 50)[0]
sections = [struct.unpack_from(">IIIIIIIIII", raw, shoff + i * shentsize) for i in range(shnum)]
names = sections[shstrndx]
fields = {2: 0xFFFFFFFF, 4: 0x03FFFFFF, 5: 0xFFFF, 6: 0xFFFF}
for index, (name, kind, flags, addr, off, size, link, info, align, entsize) in enumerate(sections):
    called = raw[names[4] + name:raw.index(0, names[4] + name)]
    if called.startswith((b".rodata", b".rdata")):
        struct.pack_into(">I", raw, shoff + index * shentsize + 8, flags & ~2)
    if kind != 9:
        continue
    target = sections[info]
    for at in range(off, off + size, 8):
        place, what = struct.unpack_from(">II", raw, at)
        mask = fields.get(what & 255)
        if mask is None:
            sys.exit("%s: relocation type %d is not one a ROM-held object keeps" % (path, what & 255))
        word = struct.unpack_from(">I", raw, target[4] + place)[0] & ~mask
        struct.pack_into(">I", raw, target[4] + place, word)
    raw[off:off + size] = bytes(size)
    struct.pack_into(">I", raw, shoff + index * shentsize + 20, 0)
open(path, "wb").write(raw)
