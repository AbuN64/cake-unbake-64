# A compiler run with its function-sections flag gives each function a section of its own (`.Fn`), but its assembler
# front rewrites the assembly and puts every function back in `.text`, every string, jump table and float pool in one
# `.rdata`. FSPLIT (text_sizes.mk: an object whose functions or read-only data the ROM places apart) puts each function
# in `.text.Fn` again and tags each read-only block with the function it belongs to, as `.rdata.Fn`: a block inside the
# function (jump table, float pool) is that function's, a block is named as the ROM names it (its first label, else its
# function's pool), so the link script lays every block at its ROM address.
import re, sys
lines = open(sys.argv[1]).read().replace("\r", "").split("\n")
code = re.compile(r"^\.section\s+\.([^\s,\"]+),\"ax\"")
reopen = re.compile(r"^\.section\s+\.([^\s,\"]+)\Z")
names = [m.group(1) for m in map(code.match, lines) if m]
funcs = set(names)
label_re = re.compile(r"^((?!RODATA_SYM_)[A-Za-z_][A-Za-z_0-9]*):")  # not the compiler front's own pool labels
def next_block(i):
    for j in range(i + 1, len(lines)):
        if lines[j].startswith(".section") or lines[j].strip() in (".text", ".data"):
            return j
    return len(lines)
def next_section(i):
    for j in range(i + 1, len(lines)):
        text = lines[j].strip()
        if text.startswith((".section", ".data")) or text == ".text":
            return lines[j]
    return ""
data = re.compile(r"\s*\.(word|half|short|byte|float|double|dword|quad|ascii|asciz|asciiz|string|space|skip)\b")
def first_align(block):
    """The `.align` a block's first item is placed by (0 when an item comes before any)."""
    for x in block:
        m = re.match(r"\s*\.align\s+(\d+)", x)
        if m:
            return int(m.group(1))
        if data.match(x):
            return 0
    return 0
# the compiler writes a pool where the previous function ends or where its own code is: the function a pool belongs to
# is the first one whose code names a label of it
label_any = re.compile(r"^([.A-Za-z_][\w.]*):")
token = re.compile(r"[.A-Za-z_][\w.]*")
owner, at, in_code = {}, None, False
for line in lines:
    found = code.match(line) or (reopen.match(line) if reopen.match(line) and reopen.match(line).group(1) in funcs else None)
    if found:
        at, in_code = found.group(1), True
    elif line.strip() == ".text":
        in_code = True
    elif re.match(r"\s*\.(section|data)\b", line):
        in_code = False
    elif in_code and at and not label_any.match(line):
        for word in token.findall(line):
            owner.setdefault(word, at)
out, current, opened = [], None, set()
for i, line in enumerate(lines):
    found = code.match(line)
    if found:
        current = found.group(1)
        out.append('.section\t.text.%s,"ax",@progbits' % current)
    elif reopen.match(line) and reopen.match(line).group(1) in funcs:
        current = reopen.match(line).group(1)
        out.append('.section\t.text.%s,"ax",@progbits' % current)
    elif line.strip() == ".text":
        out.append('.section\t.text.%s,"ax",@progbits' % current if current else line)
    elif re.match(r"^\s*\.section\s+\.(rdata|rodata)\s*\Z", line):
        base = "rdata" if "rdata" in line else "rodata"
        label = next((m.group(1) for m in map(label_re.match, lines[i + 1 : next_block(i)]) if m), None)
        # a block is named as the ROM names it: its first source label, else the pool of the function it sits in (strings before the first function: that one's)
        user = next((owner[m.group(1)] for m in map(label_any.match, lines[i + 1 : next_block(i)]) if m and m.group(1) in owner), None)
        key = label or ("literals_" + (user or current or names[0]) if names else None)
        out.append('.section\t.%s.%s,"a",@progbits' % (base, key) if key else line)
        # `SUBALIGN(1)` drops the section's alignment: a block whose first item is aligned to 8 says so to the link
        # script. A `.align 3` further in (a double after a float) pads inside the block, never moves its start, and
        # the compiler writes a pool in several pieces: only the piece that opens the block holds its first item
        if key and key not in opened and first_align(lines[i + 1 : next_block(i)]) >= 3:
            out.append("\t.weak __align8_%s\n\t.set __align8_%s, 0" % (key, key))
        opened.add(key)
    else:
        out.append(line)
open(sys.argv[2], "w").write("\n".join(out))
