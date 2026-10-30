/* hdrkey: the compile key of a preprocessed C unit (cpp -P output).
 *
 * Prints the unit with every top-level declaration it never references left out: extern variables, prototypes,
 * typedefs and struct/union/enum definitions whose names are not used, transitively, by what is kept. Function
 * and variable definitions, pragmas and anything not understood are always kept, in their order. The text is
 * compilable C (whitespace collapsed to one space where the source had any), so cc1 gives the same object for
 * it as for the full unit; two units with equal keys compile to the same object.
 * Anything this parser does not understand makes the key the whole file ("RAW" and its bytes): a key may be
 * less reduced, never wrong.
 *
 *   hdrkey FILE            key to stdout
 *   hdrkey --stats FILE    key to stdout, one line "items kept dropped" to stderr
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { T_ID, T_NUM, T_STR, T_PRAG, T_PUNCT };
typedef struct { int off, len, kind, id, sp; } Tok;
typedef struct { int a, b, root, kept, d0, nd; } Item;

static char *src; static long srclen;
static Tok *tk; static int ntk, ctk;
static Item *it; static int nit, cit;
static int *defs; static int ndefs, cdefs;
static int fallback;

/* identifier table: open addressing; kw[] marks C keywords, td[] typedef names */
typedef struct { int off, len, kw, td; } Name;
static Name *nm; static int nnm, cnm; static int *tab; static int tabn;

static const char *KW[] = {"auto","break","case","char","const","continue","default","do","double","else","enum",
 "extern","float","for","goto","if","int","long","register","return","short","signed","sizeof","static","struct",
 "switch","typedef","union","unsigned","void","volatile","while","inline","__inline","__inline__","__volatile__",
 "__const","__const__","__signed__","asm","__asm","__asm__","__attribute__","__attribute","__extension__",
 "__typeof__","typeof","restrict","__restrict","_Bool",0};

static unsigned hash(const char *s, int n) { unsigned h = 2166136261u; while (n--) h = (h ^ (unsigned char)*s++) * 16777619u; return h; }
static int intern(const char *s, int n) {
    unsigned h = hash(s, n) & (tabn - 1);
    while (tab[h] >= 0) {
        Name *x = &nm[tab[h]];
        if (x->len == n && !memcmp(src + x->off, s, n)) return tab[h];
        h = (h + 1) & (tabn - 1);
    }
    if (nnm == cnm) { cnm = cnm ? cnm * 2 : 1024; nm = realloc(nm, cnm * sizeof *nm); }
    nm[nnm].off = s - src; nm[nnm].len = n; nm[nnm].kw = 0; nm[nnm].td = 0;
    tab[h] = nnm;
    return nnm++;
}

static int dbg;
static void badat(int line, int tokidx) { if (!fallback && dbg) { fprintf(stderr, "RAW at hdrkey.c:%d token %d\n", line, tokidx); if (tokidx > 0 && tokidx < ntk) { int o = tk[tokidx].off; fprintf(stderr, "%.*s\n", 200, src + (o > 100 ? o - 100 : 0)); } } fallback = 1; }
#define bad() badat(__LINE__, i)

static int isid0(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == 36; }
static int isid(int c) { return isid0(c) || (c >= '0' && c <= '9'); }
static int isdig(int c) { return c >= '0' && c <= '9'; }

static void addtok(int off, int len, int kind, int id, int sp) {
    if (ntk == ctk) { ctk = ctk ? ctk * 2 : 4096; tk = realloc(tk, ctk * sizeof *tk); }
    tk[ntk].off = off; tk[ntk].len = len; tk[ntk].kind = kind; tk[ntk].id = id; tk[ntk].sp = sp; ntk++;
}

static void lex(void) {
    long i = 0; int sp = 0, bol = 1;
    while (i < srclen) {
        int c = (unsigned char)src[i];
        if (c == '\n') { bol = 1; sp = 1; i++; continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') { sp = 1; i++; continue; }
        if (c == '#' && bol) {               /* a directive left by cpp -P (pragma, ident): kept verbatim */
            long j = i;
            while (j < srclen && src[j] != '\n') { if (src[j] == '\\' && j + 1 < srclen) j++; j++; }
            addtok(i, j - i, T_PRAG, -1, sp); i = j; sp = 1; continue;
        }
        bol = 0;
        if (isid0(c)) {
            long j = i + 1; while (j < srclen && isid((unsigned char)src[j])) j++;
            addtok(i, j - i, T_ID, intern(src + i, j - i), sp); i = j; sp = 0; continue;
        }
        if (isdig(c) || (c == '.' && i + 1 < srclen && isdig((unsigned char)src[i + 1]))) {
            long j = i + 1;
            while (j < srclen) {
                int d = (unsigned char)src[j];
                if ((d == '+' || d == '-') && (src[j-1]=='e'||src[j-1]=='E'||src[j-1]=='p'||src[j-1]=='P')) { j++; continue; }
                if (isid(d) || d == '.') { j++; continue; }
                break;
            }
            addtok(i, j - i, T_NUM, -1, sp); i = j; sp = 0; continue;
        }
        if (c == '"' || c == '\'') {
            long j = i + 1;
            while (j < srclen && src[j] != c) { if (src[j] == '\\') j++; if (j < srclen && src[j] == '\n') { bad(); break; } j++; }
            if (j >= srclen) { bad(); return; }
            addtok(i, j + 1 - i, T_STR, -1, sp); i = j + 1; sp = 0; continue;
        }
        addtok(i, 1, T_PUNCT, -1, sp); i++; sp = 0;
    }
}

static int is_p(int i, char c) { return i >= 0 && i < ntk && tk[i].kind == T_PUNCT && src[tk[i].off] == c; }
static int is_id(int i) { return i >= 0 && i < ntk && tk[i].kind == T_ID; }
static int is_kw(int i, const char *w) {
    return is_id(i) && tk[i].len == (int)strlen(w) && !memcmp(src + tk[i].off, w, tk[i].len);
}
static int iskwid(int i) { return is_id(i) && nm[tk[i].id].kw; }

/* index of the token after the group opened at i (( [ {), or -1 when unbalanced */
static int skip_group(int i) {
    int d = 0;
    for (; i < ntk; i++) {
        if (tk[i].kind != T_PUNCT) continue;
        char c = src[tk[i].off];
        if (c == '(' || c == '[' || c == '{') d++;
        else if (c == ')' || c == ']' || c == '}') { if (--d == 0) return i + 1; if (d < 0) return -1; }
    }
    return -1;
}

static void adddef(int id) {
    if (ndefs == cdefs) { cdefs = cdefs ? cdefs * 2 : 4096; defs = realloc(defs, cdefs * sizeof *defs); }
    defs[ndefs++] = id;
}

/* Split [a,b) into declarators at depth-0 commas and collect names. Returns 0 when it cannot say what is
 * declared (the item is then kept). *allfn is set to 1 when every declarator is a function declarator. */
static int declare(int a, int b, int *allfn, int noadd) {
    int i = a, seg = 1, nfn = 0, nvar = 0;
    while (i < b) {
        /* one segment */
        int last = -1, kind = 0, j = i;     /* kind: 0 none, 1 var, 2 fn */
        while (j < b) {
            if (is_p(j, ',')) break;
            if (iskwid(j) && (is_kw(j, "struct") || is_kw(j, "union") || is_kw(j, "enum"))) {
                int k = j + 1;
                while (is_kw(k, "__attribute__")) { k = skip_group(k + 1); if (k < 0) return 0; }
                if (is_id(k) && !iskwid(k)) { if (!noadd) adddef(tk[k].id); k++; }
                if (is_p(k, '{')) {
                    int e = skip_group(k); if (e < 0) return 0;
                    if (is_kw(j, "enum")) {      /* enumerators: identifiers after { or , at brace depth 1 */
                        int d = 0;
                        for (int m = k; m < e; m++) {
                            if (is_p(m, '{') || is_p(m, '(') || is_p(m, '[')) d++;
                            else if (is_p(m, '}') || is_p(m, ')') || is_p(m, ']')) d--;
                            else if (d == 1 && is_id(m) && (is_p(m - 1, '{') || is_p(m - 1, ',')) && !noadd) adddef(tk[m].id);
                        }
                    } else {                    /* nested tags and enums inside: add their names too */
                        for (int m = k; m < e; m++)
                            if ((is_kw(m, "struct") || is_kw(m, "union") || is_kw(m, "enum")) && is_id(m + 1) && !iskwid(m + 1) && !noadd)
                                adddef(tk[m + 1].id);
                        for (int m = k; m < e; m++)
                            if (is_kw(m, "enum")) return 0;     /* enum inside a struct body: do not guess */
                    }
                    k = e;
                }
                j = k; continue;
            }
            if (is_kw(j, "__attribute__") || is_kw(j, "__attribute")) { j = skip_group(j + 1); if (j < 0) return 0; continue; }
            if (is_kw(j, "asm") || is_kw(j, "__asm") || is_kw(j, "__asm__")) return 0;
            if (is_p(j, '{')) return 0;
            if (is_id(j)) { if (!iskwid(j)) last = j; j++; continue; }
            if (is_p(j, '[')) { if (last >= 0) { kind = 1; } j = skip_group(j); if (j < 0) return 0; continue; }
            if (is_p(j, '=')) return 0;
            if (is_p(j, '(') && last == -2) { j = skip_group(j); if (j < 0) return 0; continue; }   /* params of a (*name) */
            if (is_p(j, '(')) {
                int nx = j + 1;
                if (is_p(nx, '*') || is_p(nx, '(') || is_p(nx, '^')) {      /* (*name)(...): a pointer declarator */
                    int e = skip_group(j); if (e < 0) return 0;
                    int any = 0;
                    for (int m = j + 1; m < e - 1; m++) if (is_id(m) && !iskwid(m)) { if (!noadd) adddef(tk[m].id); any = 1; }
                    if (!any) return 0;
                    kind = 1; last = -2; j = e; continue;
                }
                if (last < 0) return 0;
                int e = skip_group(j); if (e < 0) return 0;
                if (e - j == 3 && is_id(j + 1) && !iskwid(j + 1) && !nm[tk[j + 1].id].td) {
                    /* (name) after a type: a parenthesised declarator or an untyped parameter; keep both */
                    if (!noadd) { adddef(tk[last].id); adddef(tk[j + 1].id); }
                    kind = 2; last = -2; j = e; continue;
                }
                if (!noadd) adddef(tk[last].id);
                kind = 2; last = -2; j = e; continue;
            }
            j++;
        }
        if (kind == 0 && last >= 0) { kind = 1; }
        if (kind == 1 && last >= 0) { if (!noadd) adddef(tk[last].id); }
        if (kind == 0 && last < 0 && seg == 1 && j >= b) { /* specifiers only (struct S {...}; or struct S;) */ }
        else if (kind == 0) return 0;
        if (kind == 2) nfn++; else if (kind == 1) nvar++;
        i = j + 1; seg++;
    }
    *allfn = (nfn > 0 && nvar == 0);
    return 1;
}

static void classify(Item *x) {
    int a = x->a, b = x->b;
    x->d0 = ndefs; x->nd = 0; x->root = 1;
    if (!is_p(b - 1, ';')) return;          /* a function definition: always kept */
    int e = b - 1, typedef_ = 0, extern_ = 0;
    for (int i = a; i < e; i++) {
        if (tk[i].kind == T_PRAG) return;
        if (is_kw(i, "typedef")) typedef_ = 1;
        else if (is_kw(i, "extern")) extern_ = 1;
        else if (is_p(i, '{')) { int g = skip_group(i); if (g < 0) return; i = g - 1; }
    }
    if (e == a) return;                     /* a stray ; */
    int allfn = 0;
    if (!declare(a, e, &allfn, 0)) { ndefs = x->d0; return; }
    if (!typedef_ && !extern_ && !allfn) {
        /* a variable definition allocates space and is kept; a declaration of no variable (struct S {...};) is a type */
        for (int i = a; i < e; i++) {
            if (is_p(i, '{')) { i = skip_group(i) - 1; continue; }
            if (is_id(i) && !iskwid(i) && !(i > a && (is_kw(i - 1, "struct") || is_kw(i - 1, "union") || is_kw(i - 1, "enum"))))
                { ndefs = x->d0; return; }
        }
    }
    x->root = 0; x->nd = ndefs - x->d0;
}

static int bad_after_paren(int i) {      /* K&R: ) followed by an identifier that is not an attribute/asm */
    return is_id(i) && !is_kw(i, "__attribute__") && !is_kw(i, "__attribute") && !is_kw(i, "asm") && !is_kw(i, "__asm__") && !is_kw(i, "__asm");
}

static void split(void) {
    int i = 0;
    while (i < ntk && !fallback) {
        int a = i;
        if (tk[i].kind == T_PRAG) { i++; goto push; }
        int pd = 0, eq = 0;
        for (;;) {
            if (i >= ntk) { bad(); return; }
            if (tk[i].kind == T_PRAG) { i++; continue; }
            if (pd == 0 && (is_kw(i, "__attribute__") || is_kw(i, "__attribute"))) {
                int g = skip_group(i + 1); if (g < 0) { bad(); return; }
                i = g; continue;
            }
            if (tk[i].kind == T_PUNCT) {
                char c = src[tk[i].off];
                if (c == '(' || c == '[') { pd++; i++; continue; }
                if (c == ')' || c == ']') {
                    if (--pd < 0) { bad(); return; }
                    i++;
                    if (c == ')' && pd == 0 && !eq && bad_after_paren(i)) { bad(); return; }
                    continue;
                }
                if (c == '=' && pd == 0) eq = 1;
                if (c == '{' && pd == 0) {
                    int g = skip_group(i); if (g < 0) { bad(); return; }
                    int p = i - 1;
                    while (p >= a && tk[p].kind == T_PRAG) p--;
                    int agg = 0, fn = 0; (void)agg;
                    if (p >= a && is_p(p, '=')) agg = 1;
                    else if (p >= a && (is_kw(p, "struct") || is_kw(p, "union") || is_kw(p, "enum"))) agg = 1;
                    else if (p - 1 >= a && is_id(p) && (is_kw(p - 1, "struct") || is_kw(p - 1, "union") || is_kw(p - 1, "enum"))) agg = 1;
                    else if (p >= a && is_p(p, ')')) {
                        int ty = 0;
                        for (int m = a, d = 0; m < i; m++) {
                            if (is_p(m, '(') || is_p(m, '[') || is_p(m, '{')) d++;
                            else if (is_p(m, ')') || is_p(m, ']') || is_p(m, '}')) d--;
                            else if (d == 0 && is_kw(m, "typedef")) ty = 1;
                        }
                        if (ty) { bad(); return; }
                        fn = 1;
                    } else { bad(); return; }
                    i = g;
                    if (fn) { goto push; }
                    continue;
                }
                if (c == '}') { bad(); return; }
                if (c == ';' && pd == 0) { i++; goto push; }
            }
            i++;
        }
    push:
        if (nit == cit) { cit = cit ? cit * 2 : 1024; it = realloc(it, cit * sizeof *it); }
        it[nit].a = a; it[nit].b = i; it[nit].root = 1; it[nit].kept = 0; it[nit].d0 = 0; it[nit].nd = 0; nit++;
    }
}

static void emit_tokens(FILE *o, int a, int b) {
    for (int i = a; i < b; i++) {
        if (tk[i].kind == T_PRAG) { fputc('\n', o); fwrite(src + tk[i].off, 1, tk[i].len, o); fputc('\n', o); continue; }
        if (tk[i].sp && i > a) fputc(' ', o);
        fwrite(src + tk[i].off, 1, tk[i].len, o);
    }
}

int main(int argc, char **argv) {
    int stats = 0; const char *path = 0;
    for (int i = 1; i < argc; i++) { if (!strcmp(argv[i], "--stats")) stats = dbg = 1; else path = argv[i]; }
    if (!path) { fprintf(stderr, "usage: hdrkey [--stats] FILE\n"); return 2; }
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return 2; }
    fseek(f, 0, SEEK_END); srclen = ftell(f); fseek(f, 0, SEEK_SET);
    src = malloc(srclen + 1);
    if (fread(src, 1, srclen, f) != (size_t)srclen) { perror(path); return 2; }
    src[srclen] = 0; fclose(f);
    tabn = 1 << 12; while (tabn < srclen / 4) tabn <<= 1;
    tab = malloc(tabn * sizeof *tab); memset(tab, -1, tabn * sizeof *tab);
    /* keywords live in a static copy appended to the buffer so intern() can point into src */
    {
        long extra = 0; for (int i = 0; KW[i]; i++) extra += strlen(KW[i]) + 1;
        src = realloc(src, srclen + extra + 1);
        long p = srclen;
        for (int i = 0; KW[i]; i++) { int n = strlen(KW[i]); memcpy(src + p, KW[i], n); int id = intern(src + p, n); nm[id].kw = 1; p += n + 1; src[p - 1] = 0; }
    }
    long full = srclen;
    srclen = full;
    lex();
    if (!fallback) split();
    if (!fallback) {
        /* typedef names first: they tell a type in (name) apart from a declarator */
        for (int pass = 0; pass < 2; pass++) {
            ndefs = 0;
            for (int k = 0; k < nit; k++) {
                Item *x = &it[k];
                if (pass == 0) {
                    int e = x->b - 1, a = x->a, isty = 0;
                    if (!is_p(e, ';')) continue;
                    for (int i = a; i < e; i++) { if (is_kw(i, "typedef")) isty = 1; if (is_p(i, '{')) { int g = skip_group(i); if (g < 0) break; i = g - 1; } }
                    if (!isty) continue;
                    int fnx = 0, d0 = ndefs;
                    if (declare(a, e, &fnx, 0)) for (int q = d0; q < ndefs; q++) nm[defs[q]].td = 1;
                    ndefs = d0;
                } else classify(x);
            }
        }
        /* reachability: roots first, then declarations naming something live */
        int *head = malloc((nnm + 1) * sizeof *head), *nxt = malloc((ndefs + 1) * sizeof *nxt), *own = malloc((ndefs + 1) * sizeof *own);
        char *live = calloc(nnm + 1, 1);
        int *stack = malloc((ntk + nnm + 16) * sizeof *stack), sp = 0;
        for (int i = 0; i < nnm; i++) head[i] = -1;
        for (int k = 0; k < nit; k++) if (!it[k].root) for (int q = it[k].d0; q < it[k].d0 + it[k].nd; q++) { own[q] = k; nxt[q] = head[defs[q]]; head[defs[q]] = q; }
        #define KEEP(K) do { Item *y = &it[K]; y->kept = 1; for (int m = y->a; m < y->b; m++) if (tk[m].kind == T_ID && !live[tk[m].id]) { live[tk[m].id] = 1; stack[sp++] = tk[m].id; } } while (0)
        for (int k = 0; k < nit; k++) if (it[k].root) KEEP(k);
        while (sp) {
            int id = stack[--sp];
            for (int q = head[id]; q >= 0; q = nxt[q]) if (!it[own[q]].kept) KEEP(own[q]);
        }
        int kept = 0, dropped = 0;
        for (int k = 0; k < nit; k++) { if (it[k].kept) kept++; else dropped++; }
        for (int k = 0; k < nit; k++) if (it[k].kept) { emit_tokens(stdout, it[k].a, it[k].b); fputc('\n', stdout); }
        if (stats) fprintf(stderr, "%d %d %d\n", nit, kept, dropped);
        return 0;
    }
    fputs("RAW\n", stdout);
    fwrite(src, 1, full, stdout);
    if (stats) fprintf(stderr, "RAW\n");
    return 0;
}
