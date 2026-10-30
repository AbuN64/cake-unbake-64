# Make trusts file dates, and a date can lie: a tree copied, rebased or fast-forwarded gives a stamp that is newer than the
# inputs it was made from. STAMP_SWEEP runs before make reads any date. For each .i.ok stamp it hashes the inputs the
# preprocess read (the source and every header in its .d); for each object it hashes the extracted files the
# object assembles (its .i is dated by the preprocess rule, which writes it only when its bytes change) (asm.deps). The hashes seen at the last sweep are kept beside the stamp's date (build/.inputs.json).
# A stamp whose date did not move but whose inputs' bytes did was made from other bytes: it is dated long ago (make
# skips a missing stamp when the object is newer than the source), so make makes it again, and an object is removed. A stamp that was rebuilt since (its date moved) records the bytes it now has.
import hashlib, json, os, sys, time
state_path = "build/.inputs.json"
seen = {}
known = {}
cache_path = "build/.dep-stat.json"
def fingerprint(path):
    try:
        info = os.stat(path)
    except OSError:
        return [0, 0, -1]
    return [info.st_mtime_ns, info.st_ctime_ns, info.st_size]
def digest(path):
    if path not in seen:
        fp, old = fingerprint(path), known.get(path)
        if old and old[:3] == fp:
            seen[path] = old[3]
        else:
            try:
                with open(path, "rb") as handle:
                    seen[path] = hashlib.sha1(handle.read()).hexdigest()
            except OSError:
                seen[path] = "-"
        known[path] = fp + [seen[path]]
    return seen[path]
def dependencies(dfile):
    try:
        text = open(dfile).read()
    except OSError:
        return None
    names = []
    for line in text.replace(chr(92) + chr(10), " ").splitlines():
        target, colon, named = line.partition(": ")
        if colon and named.strip():
            names += named.split()
    return names
def stamp_key(stamp):
    deps = dependencies(stamp[: -len(".i.ok")] + ".d")
    return None if deps is None else hashlib.sha1(" ".join(digest(d) for d in deps).encode()).hexdigest()
def object_key(obj, extracted):
    return hashlib.sha1(" ".join(digest(a) for a in sorted(set(extracted))).encode()).hexdigest()
log_path = "build/.inputs.log"
def replay(state, path):
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    name, when, key = json.loads(line)
                except ValueError:
                    continue
                state[name] = [when, key]
    except OSError:
        pass
def load():
    try:
        with open(state_path) as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {}
    replay(state, log_path)
    return state
def save(state):
    os.makedirs("build", exist_ok=True)
    tmp = state_path + "." + str(os.getpid())
    with open(tmp, "w") as handle:
        json.dump(state, handle)
    os.replace(tmp, state_path)
if sys.argv[1] == "record":
    path, key = sys.argv[2], None
    key = stamp_key(path) if path.endswith(".i.ok") else object_key(path, sys.argv[3:])
    if key is not None:
        line = (json.dumps([path, os.stat(path).st_mtime_ns, key]) + chr(10)).encode()
        fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    sys.exit(0)
asm = {}
asm_text = ""
if os.path.exists("build/asm.deps"):
    with open("build/asm.deps") as handle:
        asm_text = handle.read()
for line in asm_text.splitlines():
    target, _, named = line.partition(": ")
    asm[target] = named.split()
versions = set(sys.argv[2:])
def twin(obj):
    parts = obj.split("/")
    return obj if parts[1] in versions else "/".join(parts[:1] + parts[2:])
def save_cache(doc):
    tmp = cache_path + "." + str(os.getpid())
    with open(tmp, "w") as handle:
        json.dump(doc, handle)
    os.replace(tmp, cache_path)
# Incremental sweep: no output's key can have changed while every file a key was made from still has the date, change
# time and size the last sweep saw (build/.dep-stat.json). Then only the outputs recorded since (build/.inputs.log) matter:
# the files they were made from join the cache, and the log stays unfolded until it is large. Anything else walks all.
sig = [sorted(versions), hashlib.sha1(asm_text.encode()).hexdigest()]
try:
    with open(cache_path) as handle:
        doc = json.load(handle)
except (OSError, ValueError):
    doc = {}
FOLD = 2000000
def moved_outputs(outs):
    for path, when in outs.items():
        try:
            if os.stat(path).st_mtime_ns != when:
                return True
        except OSError:
            return True
    return False
if (doc.get("sig") == sig and time.time() - doc.get("at", 0) < 600 and "outs" in doc
        and all(fingerprint(p) == v[:3] for p, v in doc.get("files", {}).items()) and not moved_outputs(doc["outs"])):
    known.update(doc["files"])
    try:
        size = os.stat(log_path).st_size
    except OSError:
        size = 0
    if size <= FOLD:
        done = doc.get("log", 0) if doc.get("log", 0) <= size else 0
        if size > done:
            fresh = {}
            with open(log_path, "rb") as handle:
                handle.seek(done)
                data = handle.read(size - done)
            data = data[: data.rfind(b"\n") + 1]  # a record being appended is read at the next sweep
            size = done + len(data)
            for line in data.decode(errors="replace").splitlines():
                try:
                    name, when, _ = json.loads(line)
                    fresh[name] = when
                except (ValueError, TypeError):
                    continue
            if all(os.path.exists(n) and os.stat(n).st_mtime_ns == w for n, w in fresh.items()):
                for name in fresh:
                    if name.endswith(".i.ok"):
                        for dep in dependencies(name[: -len(".i.ok")] + ".d") or []:
                            digest(dep)
                    else:
                        for dep in asm.get(twin(name), []):
                            digest(dep)
                doc["outs"].update(fresh)
                save_cache({"sig": sig, "files": known, "log": size, "outs": doc["outs"], "at": doc["at"]})
                sys.exit(0)
        else:
            sys.exit(0)
known.clear()
known.update(doc.get("files", {}) if doc.get("sig") == sig else {})
pending = log_path + "." + str(os.getpid())
try:
    os.replace(log_path, pending)
except OSError:
    pending = None
state = load()
if pending:
    replay(state, pending)
removed = 0
visited = {}
def check(path, key):
    global removed
    if key is None:
        return
    try:
        when = os.stat(path).st_mtime_ns
    except OSError:
        state.pop(path, None)
        return
    visited[path] = when
    old = state.get(path)
    if old and old[0] == when and old[1] != key:
        if path.endswith(".o"):
            os.unlink(path)
        else:
            os.utime(path, ns=(1000000000, 1000000000))
        state.pop(path, None)
        removed += 1
    else:
        state[path] = [when, key]
for here, _, names in os.walk("build"):
    for name in names:
        if name.endswith(".i.ok"):
            stamp = os.path.join(here, name)
            check(stamp, stamp_key(stamp))
            obj = stamp[: -len(".i.ok")] + ".o"
            check(obj, object_key(obj, asm.get(twin(obj), [])))
save(state)
save_cache({"sig": sig, "files": {p: known[p] for p in seen}, "log": 0, "outs": visited, "at": time.time()})
if pending:
    os.unlink(pending)
if removed:
    sys.stderr.write("stamp sweep: %d outputs made from other bytes than their inputs now hold were dropped\n" % removed)
