# Both scripts run before make reads a directory of build/: make keeps the listing it first read, so a stamp the sweep
# removes later would still look present.
# The compiler's assembly for a unit holds only an `.include` line for each extracted file its INCLUDE_ASM, INCLUDE_RODATA
# and INCLUDE_DATA lines name: the assembler reads the file's content, so an edit in place (a symbol renamed by hand or
# by a tool, with the .i unchanged and the compile answered by the cache) is an input of the object that no other rule
# lists, and the object would keep the old symbols until it is deleted. Every source's object so depends on the files
# of each version that its lines name and that exist (a line another version does not extract adds nothing).
import os, re, sys
named = re.compile(rb'INCLUDE_(ASM|RODATA|DATA)\(\s*"(\w+)"\s*,\s*(\w+)\s*[,)]')
out, lines = sys.argv[1], []
for here, _, names in sorted(os.walk("src")):
    for name in sorted(names):
        path = os.path.join(here, name)
        with open(path, "rb") as source:
            found = sorted(set(named.findall(source.read())))
        for version in sys.argv[2:]:
            files = []
            for kind, segment, symbol in found:
                folder = "nonmatchings" if kind == b"ASM" else "data"
                asm = "asm/%s/%s/%s/%s.s" % (version, folder, segment.decode(), symbol.decode())
                if os.path.exists(asm):
                    files.append(asm)
            if files:
                lines.append("build/%s/%s.o: %s\n" % (version, path, " ".join(files)))
tmp = "%s.%d" % (out, os.getpid())
with open(tmp, "w") as handle:
    handle.write("".join(lines))
os.replace(tmp, out)
