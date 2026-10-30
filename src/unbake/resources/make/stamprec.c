/* stamprec: one stamp record, appended to build/.inputs.log exactly as `stamp_sweep.py record` writes it.
 *
 * A record per output costs a python start and a hash of every header the unit read (20 ms; two per unit, so a
 * header edit that preprocesses 2000 units again spent 80 of its 117 CPU s on records). This is the same record
 * in C: the key of a preprocess stamp is the SHA-1 of the SHA-1s of the files its .d names (" "-joined, "-" for one
 * that cannot be read); the key of an object is that of the extracted files named after it, sorted and once each.
 * Exit 0 with no line when the stamp has no .d (as the python record); exit 2, before writing, for a path JSON
 * would escape, so the caller falls back to the python record.
 *
 *   stamprec record PATH [EXTRACTED...]
 *
 * The same process does what cc_cache.py does for the object rule, for the same reason (a python start is 12 ms,
 * this is under 1): `object` is the whole recipe of one X.o (cc_cache.object_file), and its PROGRAM, the compile
 * chain the toolchain's row spells, runs here without a shell: steps are separated by the word `&&` and a step is
 * a command to start or one of the file operations the chain used to start a process for (`@rotate`, `@compile`,
 * `@nonempty`, `@jumptable`, `@settle`, `@out`: cc_cache.object_file lists them).
 *
 *   stamprec object OUT VERSION FACTS CHANGED TEXTSIZE FLAGS STAMPREC [EXTRACTED...] -- PROGRAM...
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct { uint32_t h[5]; uint64_t n; unsigned char b[64]; int k; } Sha;

static uint32_t rol(uint32_t x, int s) { return (x << s) | (x >> (32 - s)); }

static void sha_block(Sha *c, const unsigned char *p) {
    uint32_t w[80], A, B, C, D, E, f, k, t;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    A = c->h[0]; B = c->h[1]; C = c->h[2]; D = c->h[3]; E = c->h[4];
    for (i = 0; i < 80; i++) {
        if (i < 20) { f = (B & C) | (~B & D); k = 0x5A827999u; }
        else if (i < 40) { f = B ^ C ^ D; k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (B & C) | (B & D) | (C & D); k = 0x8F1BBCDCu; }
        else { f = B ^ C ^ D; k = 0xCA62C1D6u; }
        t = rol(A, 5) + f + E + k + w[i];
        E = D; D = C; C = rol(B, 30); B = A; A = t;
    }
    c->h[0] += A; c->h[1] += B; c->h[2] += C; c->h[3] += D; c->h[4] += E;
}

static void sha_init(Sha *c) {
    c->h[0] = 0x67452301u; c->h[1] = 0xEFCDAB89u; c->h[2] = 0x98BADCFEu; c->h[3] = 0x10325476u; c->h[4] = 0xC3D2E1F0u;
    c->n = 0; c->k = 0;
}

static void sha_add(Sha *c, const unsigned char *p, size_t n) {
    c->n += n;
    while (n) {
        size_t room = (size_t)(64 - c->k), take = room < n ? room : n;
        memcpy(c->b + c->k, p, take);
        c->k += take; p += take; n -= take;
        if (c->k == 64) { sha_block(c, c->b); c->k = 0; }
    }
}

static void sha_hex(Sha *c, char out[41]) {
    uint64_t bits = c->n * 8;
    unsigned char pad = 0x80, zero = 0, len[8];
    int i;
    sha_add(c, &pad, 1);
    while (c->k != 56) sha_add(c, &zero, 1);
    for (i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - 8 * i));
    sha_add(c, len, 8);
    for (i = 0; i < 20; i++) sprintf(out + 2 * i, "%02x", (unsigned)(c->h[i / 4] >> (24 - 8 * (i % 4))) & 0xFF);
    out[40] = 0;
}

/* the SHA-1 of a file's bytes, or "-" when it cannot be opened or read (a directory, a missing file) */
static void digest(const char *path, char out[41]) {
    static unsigned char buf[1 << 16];
    Sha c;
    ssize_t got;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { strcpy(out, "-"); return; }
    sha_init(&c);
    while ((got = read(fd, buf, sizeof buf)) > 0) sha_add(&c, buf, (size_t)got);
    close(fd);
    if (got < 0) { strcpy(out, "-"); return; }
    sha_hex(&c, out);
}

static char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    char *text;
    long n;
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return 0; }
    text = malloc(n + 1);
    if (!text || (long)fread(text, 1, n, f) != n) { fclose(f); free(text); return 0; }
    fclose(f);
    text[n] = 0;
    *size = n;
    return text;
}

static int is_space(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\v' || ch == '\f'; }

/* the key over NAMES (in order, repeats kept): SHA-1 of their digests joined by one space */
static void key_of(char **names, int count, char out[41]) {
    Sha c;
    char d[41];
    int i;
    sha_init(&c);
    for (i = 0; i < count; i++) {
        digest(names[i], d);
        if (i) sha_add(&c, (const unsigned char *)" ", 1);
        sha_add(&c, (const unsigned char *)d, strlen(d));
    }
    sha_hex(&c, out);
}

/* the names after the first ": " of each line of a make dependency file, backslash-newlines joined (stamp_sweep.py's
   dependencies()); -1 when the file cannot be read */
static int dependencies(const char *dfile, char ***names) {
    long size = 0, i, j;
    int count = 0, cap = 64;
    char *text = slurp(dfile, &size), *line, *end, *colon, *p;
    if (!text) return -1;
    for (i = j = 0; i < size; i++) {  /* "\\\n" -> " " */
        if (text[i] == '\\' && i + 1 < size && text[i + 1] == '\n') { text[j++] = ' '; i++; }
        else text[j++] = text[i];
    }
    text[j] = 0;
    *names = malloc(cap * sizeof **names);
    for (line = text; line && *line; line = end ? end + 1 : 0) {
        end = line + strcspn(line, "\n\r");
        if (*end) *end = 0; else end = 0;
        colon = strstr(line, ": ");
        if (!colon) continue;
        for (p = colon + 2;;) {
            while (*p && is_space(*p)) p++;
            if (!*p) break;
            if (count == cap) *names = realloc(*names, (cap *= 2) * sizeof **names);
            (*names)[count++] = p;
            while (*p && !is_space(*p)) p++;
            if (*p) *p++ = 0;
        }
        if (!end) break;
    }
    return count;
}

static int by_bytes(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

/* the record of PATH: its input key and date appended to build/.inputs.log; EXTRACTED are the files an object follows */
static int record(const char *path, char **extracted, int given) {
    char key[41], line[8192], **names = 0;
    size_t n;
    int count, i, fd;
    struct stat st;
    for (i = 0; path[i]; i++)
        if ((unsigned char)path[i] < 0x20 || (unsigned char)path[i] >= 0x80 || path[i] == '"' || path[i] == '\\') return 2;
    n = strlen(path);
    if (n > 5 && !strcmp(path + n - 5, ".i.ok")) {
        char dfile[4096];
        if (n - 5 + 3 > sizeof dfile) return 2;
        memcpy(dfile, path, n - 5);
        strcpy(dfile + n - 5, ".d");
        count = dependencies(dfile, &names);
        if (count < 0) return 0;  /* no .d: no key, no record */
        key_of(names, count, key);
    } else {
        names = extracted;
        count = given;
        qsort(names, count, sizeof *names, by_bytes);
        for (i = 1, n = count ? 1 : 0; i < count; i++)
            if (strcmp(names[i], names[n - 1])) names[n++] = names[i];
        key_of(names, (int)n, key);
    }
    if (stat(path, &st)) return 1;
    i = snprintf(line, sizeof line, "[\"%s\", %lld, \"%s\"]\n", path,
                 (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec, key);
    if (i < 0 || i >= (int)sizeof line) return 2;
    fd = open("build/.inputs.log", O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) return 1;
    if (write(fd, line, (size_t)i) != i) { close(fd); return 1; }
    close(fd);
    return 0;
}

typedef struct { uint32_t h[8]; uint64_t n; unsigned char b[64]; int k; } Sha2;

static uint32_t ror(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }

static void sha2_block(Sha2 *c, const unsigned char *p) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64], v[8], t1, t2;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (; i < 64; i++)
        w[i] = (ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10)) + w[i - 7] +
               (ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 16];
    for (i = 0; i < 8; i++) v[i] = c->h[i];
    for (i = 0; i < 64; i++) {
        t1 = v[7] + (ror(v[4], 6) ^ ror(v[4], 11) ^ ror(v[4], 25)) + ((v[4] & v[5]) ^ (~v[4] & v[6])) + K[i] + w[i];
        t2 = (ror(v[0], 2) ^ ror(v[0], 13) ^ ror(v[0], 22)) + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
        v[7] = v[6]; v[6] = v[5]; v[5] = v[4]; v[4] = v[3] + t1; v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t1 + t2;
    }
    for (i = 0; i < 8; i++) c->h[i] += v[i];
}

static void sha2_init(Sha2 *c) {
    static const uint32_t H[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    memcpy(c->h, H, sizeof H);
    c->n = 0; c->k = 0;
}

static void sha2_add(Sha2 *c, const unsigned char *p, size_t n) {
    c->n += n;
    while (n) {
        size_t room = (size_t)(64 - c->k), more = room < n ? room : n;
        memcpy(c->b + c->k, p, more);
        c->k += more; p += more; n -= more;
        if (c->k == 64) { sha2_block(c, c->b); c->k = 0; }
    }
}

static void sha2_hex(Sha2 *c, char out[65]) {
    uint64_t bits = c->n * 8;
    unsigned char pad = 0x80, zero = 0, len[8];
    int i;
    sha2_add(c, &pad, 1);
    while (c->k != 56) sha2_add(c, &zero, 1);
    for (i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - 8 * i));
    sha2_add(c, len, 8);
    for (i = 0; i < 32; i++) sprintf(out + 2 * i, "%02x", (unsigned)(c->h[i / 4] >> (24 - 8 * (i % 4))) & 0xFF);
    out[64] = 0;
}

/* take COMPILER ARGS...: the batch's answer for a compile (`OUT.cc`, key `OUT.cc.key`) moved to OUT when the key is the
   SHA-256 of this command (joined by \037, then a newline) and the text it compiles, else dropped; 0 only when it was
   moved. V[0] is the compiler, V[1..N-1] its arguments. */
static int take(char **v, int n) {
    static unsigned char buf[1 << 16];
    char *out = 0, *source = 0, *key, want[65], a[4200], b[4200];
    long size = 0;
    int i, sources = 0, fd;
    ssize_t got;
    size_t len;
    Sha2 c;
    for (i = 1; i < n; i++) {
        len = strlen(v[i]);
        if (!out && !strcmp(v[i], "-o") && i + 1 < n) out = v[i + 1];
        if (len > 2 && !strcmp(v[i] + len - 2, ".i")) { source = v[i]; sources++; }
    }
    if (!out || sources != 1 || strlen(out) < 2 || strlen(out) > 4000 || strcmp(out + strlen(out) - 2, ".s")) return 1;
    fd = open(source, O_RDONLY);
    if (fd < 0) return 1;
    sha2_init(&c);
    for (i = 0; i < n; i++) {
        if (i) sha2_add(&c, (const unsigned char *)"\037", 1);
        sha2_add(&c, (const unsigned char *)v[i], strlen(v[i]));
    }
    sha2_add(&c, (const unsigned char *)"\n", 1);
    while ((got = read(fd, buf, sizeof buf)) > 0) sha2_add(&c, buf, (size_t)got);
    close(fd);
    if (got < 0) return 1;
    sha2_hex(&c, want);
    snprintf(a, sizeof a, "%s.cc.key", out);
    snprintf(b, sizeof b, "%s.cc", out);
    key = slurp(a, &size);
    while (key && size && is_space(key[size - 1])) key[--size] = 0;
    if (key && !strcmp(key, want) && !rename(b, out)) { unlink(a); free(key); return 0; }
    free(key);
    unlink(b);
    unlink(a);
    return 1;
}

#define U32(p) ((unsigned long)(p)[0] << 24 | (unsigned long)(p)[1] << 16 | (unsigned long)(p)[2] << 8 | (p)[3])
#define U16(p) ((unsigned long)(p)[0] << 8 | (p)[1])

/* the .text size of a big-endian ELF32 cut to KEEP bytes when what lies past it is zero (cc_cache.trim_text) */
static void trim_text(const char *path, unsigned long keep) {
    long size = 0, k;
    unsigned char *d = (unsigned char *)slurp(path, &size), word[4];
    unsigned long o, e, c, x, names, text = 0, start, length, at;
    int found = 0, fd;
    if (!d || size < 52) { free(d); return; }
    o = U32(d + 32); e = U16(d + 46); c = U16(d + 48); x = U16(d + 50);
    if (e < 24 || x >= c || o + c * e > (unsigned long)size) { free(d); return; }
    names = U32(d + o + x * e + 16);
    for (k = 0; k < (long)c && !found; k++) {
        at = names + U32(d + o + k * e);
        if (at < (unsigned long)size && !strcmp((char *)d + at, ".text")) { text = o + k * e; found = 1; }
    }
    if (!found) { free(d); return; }
    start = U32(d + text + 16); length = U32(d + text + 20);
    if (length > keep && start + length <= (unsigned long)size) {
        for (at = start + keep; at < start + length && !d[at]; at++) ;
        if (at == start + length) {
            word[0] = keep >> 24; word[1] = keep >> 16; word[2] = keep >> 8; word[3] = keep;
            fd = open(path, O_WRONLY);
            if (fd >= 0) { if (pwrite(fd, word, 4, text + 20) != 4) fputs("stamprec: trim failed\n", stderr); close(fd); }
        }
    }
    free(d);
}

/* a whole file written (created or replaced); 0 when it is all there */
static int put(const char *path, const char *text, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    if (write(fd, text, n) != (ssize_t)n) { close(fd); return 1; }
    return close(fd) != 0;
}

static int put_line(const char *path, const char *text) {
    size_t n = strlen(text);
    char *line = malloc(n + 2);
    int bad;
    if (!line) return 1;
    memcpy(line, text, n);
    line[n] = '\n';
    bad = put(path, line, n + 1);
    free(line);
    return bad;
}

typedef struct { const char *out, *changed, *textsize, *recorded; int stop; } Object;

static int is_file(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISREG(st.st_mode);
}

/* the exit code of a started process: its status, 128 + the signal that ended it, 1 when it was not started */
static int finish(pid_t pid) {
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) < 0) return 1;
    if (!WIFEXITED(status)) return 128 + WTERMSIG(status);
    return WEXITSTATUS(status);
}

/* ARGV (null-ended) as a process, its output in INTO when given */
static int spawn(char **argv, const char *into) {
    pid_t pid = fork();
    if (pid == 0) {
        if (into) {
            int fd = open(into, O_WRONLY | O_CREAT | O_TRUNC, 0666);
            if (fd < 0 || dup2(fd, 1) < 0) _exit(126);
            close(fd);
        }
        execvp(argv[0], argv);
        fprintf(stderr, "stamprec: cannot run %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    return finish(pid);
}

/* the compile command `sh .../cc_front.sh CACHE COMPILER ARGS...` is the front: while the build only collects, its
   command is recorded and failed (125); else the batch's answer is moved into place; a unit the batch could not answer
   is the front's own run (the cache, then the compiler). Any other command is just started. */
static int compile(char **w) {
    int n = 0;
    size_t k;
    const char *collect = getenv("ABU_CC_COLLECT");
    while (w[n]) n++;
    if (n >= 5 && !strcmp(w[0], "sh") && (k = strlen(w[1])) >= 11 && !strcmp(w[1] + k - 11, "cc_front.sh")) {
        if (collect && *collect) {
            size_t size = 2;
            char *line, *at;
            int fd, i, bad;
            for (i = 3; i < n; i++) size += strlen(w[i]) + 1;
            line = at = malloc(size);
            if (!line) return 1;
            for (i = 3; i < n; i++) {
                if (i > 3) *at++ = '\037';
                at = stpcpy(at, w[i]);
            }
            *at++ = '\n';
            fd = open(collect, O_WRONLY | O_APPEND | O_CREAT, 0666);  /* one write: lines this short never interleave */
            bad = fd < 0 || write(fd, line, (size_t)(at - line)) != (ssize_t)(at - line);
            if (fd >= 0) close(fd);
            free(line);
            return bad ? 1 : 125;
        }
        if (!take(w + 3, n - 3)) return 0;
    }
    return spawn(w, 0);
}

/* a jump table entry `.word .L12` of the assembly holds its label's address less BASE (cc_cache.jump_table) */
static int jump_table(const char *path, const char *base) {
    long size = 0, lines = 0, i, from = 0, to = 0, at, digits;
    char *d = slurp(path, &size), *o;
    size_t add = strlen(base) + 4;
    int changed = 0, bad;
    if (!d) return 1;
    for (i = 0; i < size; i++) lines += d[i] == '\n';
    o = malloc((size_t)size + (size_t)(lines + 1) * add + 1);
    if (!o) { free(d); return 1; }
    for (i = 0; i <= size; i++) {
        long end, p, q;
        if (i < size && d[i] != '\n') continue;
        end = i;
        p = from;
        while (p < end && (d[p] == ' ' || d[p] == '\t')) p++;
        if (end - p >= 5 && !memcmp(d + p, ".word", 5)) {
            p += 5;
            q = p;
            while (q < end && (d[q] == ' ' || d[q] == '\t')) q++;
            if (q > p && q + 1 < end && (d[q] == '.' || d[q] == '$') && d[q + 1] == 'L') {
                at = q += 2;
                for (digits = 0; q < end && d[q] >= '0' && d[q] <= '9'; q++) digits++;
                at = q;
                while (q < end && (d[q] == ' ' || d[q] == '\t')) q++;
                if (digits && q == end) {
                    memcpy(o + to, d + from, (size_t)(at - from));
                    to += at - from;
                    to += sprintf(o + to, "-0x%s", base);
                    changed = 1;
                    goto line_done;
                }
            }
        }
        memcpy(o + to, d + from, (size_t)(end - from));
        to += end - from;
    line_done:
        if (i < size) o[to++] = '\n';
        from = i + 1;
    }
    bad = changed ? put(path, o, (size_t)to) : 0;
    free(d);
    free(o);
    return bad;
}

/* settle ASSEMBLY KEPT: the object stands as it was when only its .i moved, its size is the wanted one and the compiler's
   assembly is the kept one's, byte for byte (the rest of the chain is skipped and `.same` tells the rule its trim is held) */
static int settle(Object *x, const char *assembly, const char *kept) {
    char *a = 0, *b = 0, flag[4200];
    long na = 0, nb = 0;
    int same = 0;
    if (!*x->changed && !strcmp(x->recorded, x->textsize) && is_file(x->out) && is_file(kept)) {
        a = slurp(assembly, &na);
        b = slurp(kept, &nb);
        same = a && b && na == nb && !memcmp(a, b, (size_t)na);
    }
    free(a);
    free(b);
    unlink(kept);
    if (!same) return 0;
    snprintf(flag, sizeof flag, "%s.same", x->out);
    x->stop = 1;
    return utimensat(AT_FDCWD, x->out, 0, 0) || put(flag, "", 0);
}

/* one step of the program (cc_cache.object_file describes them) */
static int step(Object *x, char **w) {
    int n = 0;
    while (w[n]) n++;
    if (!n) return 0;
    if (!strcmp(w[0], "@rotate") && n == 3) {
        if (is_file(w[2])) unlink(w[1]);
        else if (is_file(w[1]) && rename(w[1], w[2])) return 1;
        return 0;
    }
    if (!strcmp(w[0], "@nonempty") && n == 2) {
        struct stat st;
        return stat(w[1], &st) || st.st_size <= 0;
    }
    if (!strcmp(w[0], "@jumptable") && n == 3) return jump_table(w[1], w[2]);
    if (!strcmp(w[0], "@settle") && n == 3) return settle(x, w[1], w[2]);
    if (!strcmp(w[0], "@compile") && n >= 2) return compile(w + 1);
    if (!strcmp(w[0], "@out") && n >= 3) return spawn(w + 2, w[1]);
    if (w[0][0] == '@') { fprintf(stderr, "stamprec: no step %s\n", w[0]); return 64; }
    return spawn(w, 0);
}

/* object OUT VERSION FACTS CHANGED TEXTSIZE FLAGS STAMPREC [EXTRACTED...] -- PROGRAM...: cc_cache.object_file, step for step */
static int object(int argc, char **argv) {
    char *out = argv[2], *version = argv[3], *facts = argv[4], *changed = argv[5], *textsize = argv[6];
    char *flags = argv[7];
    char path[4200], source[4200], made[128], total[64], hex[41], *old, *recorded;
    static unsigned char buf[1 << 16];
    struct stat si, so;
    Sha c;
    Object x;
    long size = 0, n;
    int i, stands = 0, program = 9, extracted;
    while (program < argc && strcmp(argv[program], "--")) program++;
    if (program >= argc - 1) return 64;
    extracted = program - 9;
    if (strlen(out) < 3 || strlen(out) > 4000 || strlen(version) > 100) return 64;
    snprintf(path, sizeof path, "asm/%s/.extracted", version);
    old = slurp(path, &size);
    snprintf(made, sizeof made, "%s", old ? old : "");
    free(old);
    for (n = (long)strlen(made); n && made[n - 1] == '\n';) made[--n] = 0;
    if (made[0] && strcmp(made, facts)) {
        fprintf(stderr, "asm/%s was extracted from other symbol_addrs.txt/splat.yaml than this tree holds (%s, now %s): run make extract-%s\n",
                version, made, facts, version);
        return 1;
    }
    sha_init(&c);
    for (i = 9; i < program; i++) {  /* the sum of the extracted files an object assembles; one that is not there adds nothing */
        ssize_t got;
        int fd = open(argv[i], O_RDONLY);
        if (fd < 0) continue;
        while ((got = read(fd, buf, sizeof buf)) > 0) sha_add(&c, buf, (size_t)got);
        close(fd);
    }
    sha_hex(&c, hex);
    snprintf(total, sizeof total, "%s  -\n", hex);
    snprintf(path, sizeof path, "%s.asm", out);
    old = slurp(path, &size);
    if (old && (size != (long)strlen(total) || memcmp(old, total, size))) unlink(out);  /* other extracted bytes: remade */
    free(old);
    snprintf(path, sizeof path, "%s.textsize", out);
    recorded = slurp(path, &size);
    if (recorded && size && recorded[size - 1] == '\n') recorded[--size] = 0;
    snprintf(source, sizeof source, "%s", out);
    source[strlen(source) - 1] = 'i';  /* X.o -> X.i */
    if (!*changed && !strcmp(recorded ? recorded : "", textsize) && !stat(source, &si) && !stat(out, &so))
        stands = si.st_mtim.tv_sec < so.st_mtim.tv_sec ||
                 (si.st_mtim.tv_sec == so.st_mtim.tv_sec && si.st_mtim.tv_nsec < so.st_mtim.tv_nsec);
    snprintf(path, sizeof path, "%s.flags", out);
    if (stands) {
        free(recorded);
        if (put_line(path, flags) || utimensat(AT_FDCWD, out, 0, 0)) return 1;
    } else {
        int code = 0;
        x.out = out; x.changed = changed; x.textsize = textsize; x.recorded = recorded ? recorded : ""; x.stop = 0;
        for (i = program + 1; i < argc && !code && !x.stop;) {  /* the steps, split at `&&` */
            int j = i;
            while (j < argc && strcmp(argv[j], "&&")) j++;
            argv[j] = 0;
            code = step(&x, argv + i);
            i = j + 1;
        }
        free(recorded);
        if (code) return code;
        snprintf(source, sizeof source, "%s.same", out);
        if (!unlink(source)) {  /* the compile's own check: the object stands as it was, its trim is held */
        } else {
            if (*textsize) trim_text(out, strtoul(textsize, 0, 0));
            snprintf(source, sizeof source, "%s.textsize", out);
            if (put_line(source, textsize)) return 1;
        }
        if (put_line(path, flags)) return 1;
    }
    snprintf(path, sizeof path, "%s.asm", out);
    if (put(path, total, strlen(total))) return 1;
    return record(out, argv + 9, extracted);
}

int main(int argc, char **argv) {
    if (argc >= 3 && !strcmp(argv[1], "record")) return record(argv[2], argv + 3, argc - 3);
    if (argc >= 11 && !strcmp(argv[1], "object")) return object(argc, argv);
    fprintf(stderr, "usage: stamprec record PATH [EXTRACTED...] | object OUT VERSION FACTS CHANGED TEXTSIZE FLAGS STAMPREC [EXTRACTED...] -- PROGRAM...\n");
    return 64;
}
