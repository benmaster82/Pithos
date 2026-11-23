/* pcache.c - prompt prefix reuse: checkpoints of the recurrent state + sessions stored on disk. */
#include "pcache.h"
#include <stdlib.h>
#include <string.h>

#define MAX_CK 64
/* Qwen's recurrent state is ~63 MB a checkpoint (LFM's a few hundred KB, Inkling's up to ~170 MB
 * with its window rows): at most this many bytes of them in memory, and in a stored session
 * those that fit in DISK_CK_BYTES, the last one first, then the first ones and the ones before
 * the last (a new conversation reuses the system prompt and the tools, a continued one the end) */
#define CK_BYTES (512ull << 20)
#define DISK_CK_BYTES (256ull << 20)
#define FILE_EXT ".pithoskv"

typedef struct {
    char name[32];
    int n, nck;
    int *toks, *ckpos;      /* ckpos ascending, last one == n */
    uint64_t seq, bytes;
} Stored;

typedef struct {
    char magic[8];
    uint64_t model_id;
    int32_t n, nck;
    uint64_t conv_bytes, kv_row_bytes, seq;
} FileHdr;

static const char MAGIC[8] = { 'L', 'F', 'M', 'C', 'K', 'V', '1', 0 };
/* checkpoints of different sizes (Inkling): their sizes (uint64) follow the positions */
static const char MAGIC2[8] = { 'L', 'F', 'M', 'C', 'K', 'V', '2', 0 };

struct PCache {
    Context *c;
    char dir[1024];
    int use_disk;
    uint64_t model_id;
    int max_files;
    uint64_t max_bytes;
    /* active session: what the context holds now */
    int *toks, n, cap;
    CtxSnapshot ck[MAX_CK];
    int nck, max_ck;
    int src;                /* stored session this one continues (overwritten on save), -1 = new */
    /* stored sessions */
    Stored *files;
    int nfiles, fcap, next_id;
    uint64_t seq;
};

static int lcp(const int *a, int na, const int *b, int nb) {
    int n = na < nb ? na : nb, i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

static void path_of(const PCache *pc, const char *name, char *out, size_t cap) {
    snprintf(out, cap, "%s/%s", pc->dir, name);
}

static void stored_free(Stored *s) {
    free(s->toks);
    free(s->ckpos);
    memset(s, 0, sizeof(*s));
}

static void drop_ck_after(PCache *pc, int pos) {
    while (pc->nck > 0 && pc->ck[pc->nck - 1].pos > pos) ctx_snapshot_free(&pc->ck[--pc->nck]);
}

/* ------------------------------------------------------------------ disk index */

static int read_index_entry(PCache *pc, const char *name, Stored *s) {
    char path[1100];
    path_of(pc, name, path, sizeof(path));
    FILE *f = file_open(path, "rb");
    if (!f) return 0;
    FileHdr h;
    int ok = fread(&h, sizeof(h), 1, f) == 1 && (!memcmp(h.magic, MAGIC, 8) || !memcmp(h.magic, MAGIC2, 8)) &&
             h.model_id == pc->model_id &&
             h.n > 0 && h.nck > 0 && h.nck <= MAX_CK && h.n <= ctx_n_ctx(pc->c) &&
             h.conv_bytes == ctx_conv_bytes(pc->c) && h.kv_row_bytes == ctx_kv_bytes(pc->c, 1);
    if (ok) {
        memset(s, 0, sizeof(*s));
        snprintf(s->name, sizeof(s->name), "%s", name);
        s->n = h.n;
        s->nck = h.nck;
        s->seq = h.seq;
        s->toks = (int *)malloc(sizeof(int) * (size_t)h.n);
        s->ckpos = (int *)malloc(sizeof(int) * (size_t)h.nck);
        ok = fread(s->toks, sizeof(int), (size_t)h.n, f) == (size_t)h.n &&
             fread(s->ckpos, sizeof(int), (size_t)h.nck, f) == (size_t)h.nck && s->ckpos[h.nck - 1] == h.n;
        uint64_t ckb = h.conv_bytes * h.nck, sz[MAX_CK];
        if (ok && !memcmp(h.magic, MAGIC2, 8)) {
            ok = fread(sz, sizeof(uint64_t), (size_t)h.nck, f) == (size_t)h.nck;
            ckb = sizeof(uint64_t) * (uint64_t)h.nck;
            for (int k = 0; ok && k < h.nck; k++) ckb += sz[k];
        }
        s->bytes = sizeof(h) + sizeof(int) * (uint64_t)(h.n + h.nck) + ckb + h.kv_row_bytes * h.n;
        if (!ok) stored_free(s);
    }
    fclose(f);
    return ok;
}

static void scan_cb(const char *name, void *u) {
    PCache *pc = (PCache *)u;
    size_t l = strlen(name), el = strlen(FILE_EXT);
    if (l <= el || l >= 32 || strcmp(name + l - el, FILE_EXT)) return;
    int id = atoi(name + 1);
    if (id >= pc->next_id) pc->next_id = id + 1;
    Stored s;
    if (!read_index_entry(pc, name, &s)) {
        char path[1100];
        path_of(pc, name, path, sizeof(path));
        file_remove(path); /* other model / damaged */
        return;
    }
    if (pc->nfiles == pc->fcap) {
        pc->fcap = pc->fcap ? pc->fcap * 2 : 16;
        pc->files = (Stored *)realloc(pc->files, sizeof(Stored) * (size_t)pc->fcap);
    }
    pc->files[pc->nfiles++] = s;
    if (s.seq > pc->seq) pc->seq = s.seq;
}

static void remove_stored(PCache *pc, int i) {
    char path[1100];
    path_of(pc, pc->files[i].name, path, sizeof(path));
    file_remove(path);
    stored_free(&pc->files[i]);
    pc->files[i] = pc->files[--pc->nfiles];
    if (pc->src == pc->nfiles) pc->src = i;
    else if (pc->src == i) pc->src = -1;
}

static void evict(PCache *pc) {
    for (;;) {
        uint64_t total = 0;
        int victim = -1;
        for (int i = 0; i < pc->nfiles; i++) {
            total += pc->files[i].bytes;
            if (i != pc->src && (victim < 0 || pc->files[i].seq < pc->files[victim].seq)) victim = i;
        }
        if (victim < 0 || (pc->nfiles <= pc->max_files && total <= pc->max_bytes)) return;
        remove_stored(pc, victim);
    }
}

PCache *pcache_create(Context *c, const char *dir, uint64_t model_id, int max_files, uint64_t max_bytes) {
    PCache *pc = (PCache *)calloc(1, sizeof(PCache));
    pc->c = c;
    pc->model_id = model_id;
    pc->max_files = max_files > 0 ? max_files : 1;
    pc->max_bytes = max_bytes;
    pc->src = -1;
    pc->next_id = 1;
    const uint64_t cb = ctx_conv_bytes(c);
    pc->max_ck = cb && CK_BYTES / cb < MAX_CK ? (int)(CK_BYTES / cb) : MAX_CK;
    if (pc->max_ck < 4) pc->max_ck = 4;
    if (dir && *dir && dir_create(dir)) {
        snprintf(pc->dir, sizeof(pc->dir), "%s", dir);
        pc->use_disk = 1;
        dir_list(dir, scan_cb, pc);
        evict(pc);
    }
    ctx_reset(c);
    return pc;
}

void pcache_free(PCache *pc) {
    if (!pc) return;
    drop_ck_after(pc, -1);
    for (int i = 0; i < pc->nfiles; i++) stored_free(&pc->files[i]);
    free(pc->files);
    free(pc->toks);
    free(pc);
}

int pcache_n_files(const PCache *pc) { return pc->nfiles; }

/* ------------------------------------------------------------------ active session */

static void set_tokens(PCache *pc, const int *t, int n) {
    if (n > pc->cap) {
        pc->cap = n * 2 + 256;
        pc->toks = (int *)realloc(pc->toks, sizeof(int) * (size_t)pc->cap);
    }
    if (t != pc->toks) memcpy(pc->toks, t, sizeof(int) * (size_t)n);
    pc->n = n;
}

void pcache_clear(PCache *pc) {
    drop_ck_after(pc, -1);
    pc->n = 0;
    pc->src = -1;
    ctx_reset(pc->c);
}

/* Rewind to pos, which is the current end or a checkpoint position (or 0). */
static void rewind_to(PCache *pc, int pos) {
    if (pos == pc->n) return;
    drop_ck_after(pc, pos);
    if (pos == 0) {
        ctx_reset(pc->c);
    } else {
        ctx_restore(pc->c, &pc->ck[pc->nck - 1]); /* its pos == pos by construction */
    }
    pc->n = pos;
}

static int best_ck(const int *pos, int n, int limit) {
    int b = 0;
    for (int i = 0; i < n; i++)
        if (pos[i] <= limit && pos[i] > b) b = pos[i];
    return b;
}

static int keep_from(const Stored *s) { return s->nck >= 2 ? s->ckpos[s->nck - 2] : s->n; }

/* Replace the active session with stored session i, truncated at checkpoint `upto`. */
static int load_stored(PCache *pc, int i, int upto) {
    Stored *s = &pc->files[i];
    char path[1100];
    path_of(pc, s->name, path, sizeof(path));
    FILE *f = file_open(path, "rb");
    if (!f) return 0;
    FileHdr h;
    int ok = fread(&h, sizeof(h), 1, f) == 1 && h.n == s->n && h.nck == s->nck;
    ok = ok && !fseek(f, (long)(sizeof(int) * (size_t)(h.n + h.nck)), SEEK_CUR);
    uint64_t sz[MAX_CK];
    for (int k = 0; k < MAX_CK; k++) sz[k] = h.conv_bytes;
    if (ok && !memcmp(h.magic, MAGIC2, 8)) ok = fread(sz, sizeof(uint64_t), (size_t)h.nck, f) == (size_t)h.nck;
    drop_ck_after(pc, -1);
    pc->n = 0;
    for (int k = 0; ok && k < s->nck; k++) {
        if (s->ckpos[k] > upto) {
            ok = !fseek(f, (long)sz[k], SEEK_CUR);
            continue;
        }
        CtxSnapshot *cs = &pc->ck[pc->nck];
        cs->pos = s->ckpos[k];
        cs->bytes = (size_t)sz[k];
        cs->conv = (float *)malloc(cs->bytes);
        pc->nck++;
        ok = cs->conv && fread(cs->conv, 1, cs->bytes, f) == cs->bytes;
    }
    ok = ok && pc->nck > 0 && pc->ck[pc->nck - 1].pos == upto && ctx_kv_read(pc->c, upto, h.n, f);
    fclose(f);
    if (!ok) { pcache_clear(pc); return 0; }
    set_tokens(pc, s->toks, upto);
    ctx_restore(pc->c, &pc->ck[pc->nck - 1]);
    pc->src = upto >= keep_from(s) ? i : -1;
    s->seq = ++pc->seq;
    return 1;
}

int pcache_prepare(PCache *pc, const int *toks, int n, int allow_full) {
    const int lim = allow_full ? n : n - 1;
    const int l = lcp(pc->toks, pc->n, toks, n);
    int best = 0;
    if (l == pc->n && pc->n <= lim) {
        best = pc->n;
    } else {
        for (int k = 0; k < pc->nck; k++)
            if (pc->ck[k].pos <= l && pc->ck[k].pos <= lim && pc->ck[k].pos > best) best = pc->ck[k].pos;
    }
    /* a stored session may share a longer prefix (another conversation, or before a restart) */
    int bf = -1, best_f = 0;
    for (int i = 0; i < pc->nfiles; i++) {
        const Stored *s = &pc->files[i];
        int lf = lcp(s->toks, s->n, toks, n);
        int b = best_ck(s->ckpos, s->nck, lf < lim ? lf : lim);
        if (b > best_f) { best_f = b; bf = i; }
    }
    if (bf >= 0 && best_f > best + 8) {
        if (load_stored(pc, bf, best_f)) return best_f;
        return 0; /* context was cleared */
    }
    rewind_to(pc, best);
    if (pc->src >= 0 && best < keep_from(&pc->files[pc->src])) pc->src = -1; /* a different branch */
    return best;
}

float *pcache_feed(PCache *pc, const int *toks, int n) {
    float *lg = ctx_forward(pc->c, toks, n);
    if (!lg) {
        if (ctx_pos(pc->c) < pc->n) pcache_clear(pc);   /* the context lost its state (GPU failure) */
        return NULL;
    }
    if (pc->n + n > pc->cap) {
        pc->cap = (pc->n + n) * 2 + 256;
        pc->toks = (int *)realloc(pc->toks, sizeof(int) * (size_t)pc->cap);
    }
    memcpy(pc->toks + pc->n, toks, sizeof(int) * (size_t)n);
    pc->n += n;
    return lg;
}

/* drop the checkpoint whose removal leaves the smallest gap (keep the latest one) */
static void drop_one_ck(PCache *pc) {
    int vi = 0, vgap = 0;
    for (int i = 0; i < pc->nck - 1; i++) {
        int gap = pc->ck[i + 1].pos - (i ? pc->ck[i - 1].pos : 0);
        if (i == 0 || gap < vgap) { vi = i; vgap = gap; }
    }
    ctx_snapshot_free(&pc->ck[vi]);
    memmove(&pc->ck[vi], &pc->ck[vi + 1], sizeof(CtxSnapshot) * (size_t)(pc->nck - 1 - vi));
    pc->nck--;
}

void pcache_checkpoint(PCache *pc) {
    if (pc->n == 0 || (pc->nck > 0 && pc->ck[pc->nck - 1].pos == pc->n)) return;
    if (pc->nck == pc->max_ck) drop_one_ck(pc);
    ctx_snapshot(pc->c, &pc->ck[pc->nck++]);
    for (;;) {   /* within CK_BYTES (Inkling's snapshots grow with the position) */
        uint64_t tot = 0;
        for (int i = 0; i < pc->nck; i++) tot += pc->ck[i].bytes;
        if (tot <= CK_BYTES || pc->nck <= 1) break;
        drop_one_ck(pc);
    }
}

int pcache_save(PCache *pc) {
    if (!pc->use_disk || pc->n == 0) return 0;
    pcache_checkpoint(pc);
    Stored s;
    memset(&s, 0, sizeof(s));
    if (pc->src >= 0) snprintf(s.name, sizeof(s.name), "%s", pc->files[pc->src].name);
    else snprintf(s.name, sizeof(s.name), "s%06d" FILE_EXT, pc->next_id++);
    char path[1100], tmp[1110];
    path_of(pc, s.name, path, sizeof(path));
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FileHdr h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, MAGIC, 8);
    h.model_id = pc->model_id;
    h.n = pc->n;
    h.conv_bytes = ctx_conv_bytes(pc->c);
    /* the checkpoints written: the last, then by turns the first ones and the ones before the
     * last, while they fit in DISK_CK_BYTES */
    int sel[MAX_CK], ns = 0, keep[MAX_CK] = {0}, varied = 0;
    uint64_t tot = 0, ckb = 0;
    for (int k = 0; k < pc->nck; k++) varied |= pc->ck[k].bytes != h.conv_bytes;
    for (int r = 0; r < pc->nck; r++) {
        const int k = r == 0 ? pc->nck - 1 : r % 2 ? (r - 1) / 2 : pc->nck - 1 - r / 2;
        if (keep[k] || (r && tot + pc->ck[k].bytes > DISK_CK_BYTES)) continue;
        keep[k] = 1;
        tot += pc->ck[k].bytes;
    }
    for (int k = 0; k < pc->nck; k++)
        if (keep[k]) { sel[ns++] = k; ckb += pc->ck[k].bytes; }
    if (varied) memcpy(h.magic, MAGIC2, 8);
    h.nck = ns;
    h.kv_row_bytes = ctx_kv_bytes(pc->c, 1);
    h.seq = ++pc->seq;
    FILE *f = file_open(tmp, "wb");
    if (!f) return 0;
    int ok = fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(pc->toks, sizeof(int), (size_t)pc->n, f) == (size_t)pc->n;
    for (int k = 0; ok && k < ns; k++) ok = fwrite(&pc->ck[sel[k]].pos, sizeof(int), 1, f) == 1;
    for (int k = 0; ok && varied && k < ns; k++) {
        const uint64_t b = pc->ck[sel[k]].bytes;
        ok = fwrite(&b, sizeof(b), 1, f) == 1;
    }
    for (int k = 0; ok && k < ns; k++) ok = fwrite(pc->ck[sel[k]].conv, 1, pc->ck[sel[k]].bytes, f) == pc->ck[sel[k]].bytes;
    ok = ok && ctx_kv_write(pc->c, pc->n, f);
    ok = (fclose(f) == 0) && ok;
    if (!ok || !file_rename(tmp, path)) { file_remove(tmp); return 0; }

    s.n = pc->n;
    s.nck = ns;
    s.seq = h.seq;
    s.toks = (int *)malloc(sizeof(int) * (size_t)s.n);
    memcpy(s.toks, pc->toks, sizeof(int) * (size_t)s.n);
    s.ckpos = (int *)malloc(sizeof(int) * (size_t)s.nck);
    for (int k = 0; k < s.nck; k++) s.ckpos[k] = pc->ck[sel[k]].pos;
    s.bytes = sizeof(h) + sizeof(int) * (uint64_t)(s.n + s.nck) + (varied ? sizeof(uint64_t) * (uint64_t)s.nck : 0) + ckb +
              h.kv_row_bytes * s.n;
    if (pc->src >= 0) {
        stored_free(&pc->files[pc->src]);
        pc->files[pc->src] = s;
    } else {
        if (pc->nfiles == pc->fcap) {
            pc->fcap = pc->fcap ? pc->fcap * 2 : 16;
            pc->files = (Stored *)realloc(pc->files, sizeof(Stored) * (size_t)pc->fcap);
        }
        pc->src = pc->nfiles;
        pc->files[pc->nfiles++] = s;
    }
    evict(pc);
    return 1;
}
