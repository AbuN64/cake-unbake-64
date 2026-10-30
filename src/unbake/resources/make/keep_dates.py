# splat writes every extracted file again, so each new date would remake every object of the version (through the
# extraction stamp and the files a source names) although nothing changed. A file whose bytes are the same gets its old
# date back. A changed or added file reaches exactly the objects whose lines name it (build/asm.deps), so the stamp
# keeps its date unless the shared asm/<v>/include changed; an object that named a file the extraction no longer
# makes (asm.deps as make read it before the extraction) is dated long ago, so make compiles it and says what it lacks.
import hashlib, json, os, sys
mode, version = sys.argv[1], sys.argv[2]
root, keep = "asm/" + version, "build/.extract-dates-" + version + ".json"
stamp = root + "/.extracted"
def scan():
    found = {}
    for here, _, names in os.walk(root):
        for name in names:
            path = os.path.join(here, name)
            if path != stamp:
                with open(path, "rb") as data:
                    found[path] = hashlib.sha1(data.read()).hexdigest()
    return found
if mode == "save":
    os.makedirs("build", exist_ok=True)
    files = {path: [digest, os.stat(path).st_mtime_ns] for path, digest in scan().items()}
    when = os.stat(stamp).st_mtime_ns if os.path.exists(stamp) else None
    with open(keep, "w") as out:
        json.dump({"files": files, "stamp": when}, out)
else:
    with open(keep) as saved:
        old = json.load(saved)
    files = scan()
    for path, digest in files.items():
        if old["files"].get(path, [None])[0] == digest:
            os.utime(path, ns=(old["files"][path][1], old["files"][path][1]))
    moved = set(files) ^ set(old["files"]) | {path for path in files if old["files"].get(path, [None])[0] != files[path]}
    if old["stamp"] is not None and not any(path.startswith(root + "/include/") for path in moved):
        os.utime(stamp, ns=(old["stamp"], old["stamp"]))
    gone = set(old["files"]) - set(files)
    if gone and os.path.exists("build/asm.deps"):
        for line in open("build/asm.deps"):
            target, _, named = line.partition(": ")
            if gone.intersection(named.split()) and os.path.exists(target):
                os.utime(target, ns=(1000000000, 1000000000))
    os.remove(keep)
