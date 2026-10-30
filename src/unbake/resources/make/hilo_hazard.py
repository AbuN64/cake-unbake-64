# The R4000 reads HI/LO late: an mfhi/mflo needs two instructions before the next mult/div/mthi/mtlo. The assembler the
# ROMs were made with wrote two nops where the compiler placed the two back to back (gas lays the gap itself only when
# one instruction is already between, so here they stay adjacent). A toolchain row's hilo_hazard_nops puts those two
# nops into the compiler's assembly, only in `.set reorder` text, where the assembler owns the slots.
import re, sys
path, out, last, reorder = sys.argv[1], [], None, True
for line in open(path):
    text = line.split("#", 1)[0].strip()
    if text.startswith(".set"):
        if re.match(r"\.set\s+noreorder\b", text):
            reorder = False
        elif re.match(r"\.set\s+reorder\b", text):
            reorder = True
    elif text and not text.startswith(".") and not re.match(r"[\w.\x24]+:", text):
        word = text.split(None, 1)[0]
        if word in ("mult", "multu", "dmult", "dmultu", "div", "divu", "ddiv", "ddivu", "mthi", "mtlo"):
            if reorder and last == 0 and (len(text.split(",")) < 3 or word.startswith("m")):
                out.append("\tnop\n\tnop\n")
        last = 0 if word in ("mfhi", "mflo") else (None if last is None else last + 1)
    out.append(line)
open(path, "w").write("".join(out))
