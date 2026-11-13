/* speech.c - speech output of LFM3-Audio (see speech.h).
 *
 * Depth decoder: the hidden state is projected to 12 positions of 1024, one per codebook, each plus
 * the embedding of its code once known, or a mask embedding. A 3-block LFM3 with attention over all
 * 12 positions (model.c, bidir) scores the codebooks still masked against their code embeddings
 * (the head is tied). MaskGIT: the codebooks, in a random order, are revealed 4 at a time over 3
 * passes. The frame goes back to the language model as the sum of its codes' embeddings.
 *
 * LiquidCodec decoder: each code is 4 FSQ values, the 48 of a frame go to 256 then 512; each frame
 * is repeated over 6 positions of an 8-block causal LFM3 (sliding window 30), whose output gives the
 * log magnitude and the phase of 641 bins per position: an inverse STFT of 1280 samples every 320
 * (Hann window, overlap-add divided by the window's squared sum, 480 samples trimmed at each end). */
#include "speech.h"
#include "model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NCB SPEECH_CODEBOOKS
#define NCODE 2017       /* codes per codebook; the last one ends the audio */
#define EOS_CODE (NCODE - 1)
#define NLVL 4           /* FSQ values per code */
#define FSTACK 6         /* codec positions per frame */
#define NFFT 1280
#define HOPS 320
#define NBIN (NFFT / 2 + 1)
#define TRIM ((NFFT - HOPS) / 2)
#define WIN 30           /* the codec's attention window */

struct Speech {
    Model depth, codec;
    Context *dc, *cc;
    ThreadPool *pool;
    int lm_dim, D, CD, max_pos;   /* D: depth decoder width, CD: codec width */
    Mat proj_in, embd, out_embd, in_proj, out_proj;
    const float *proj_in_b, *mask, *head_norm, *fsq, *po_w, *po_b, *in_proj_b, *out_proj_b;
    int *hist, n_hist;              /* [frame][NCB] codes so far (repetition penalty) */
    float win[NFFT], cosv[NFFT], sinv[NFFT];
    float ola[NFFT], env[NFFT];     /* sums of the frames over the samples not yet complete */
    int skip;                       /* samples still to drop at the start */
    float *hp, *x, *y, *logits, *pos, *spec, *frames;   /* frames: [FSTACK][NFFT] inverse FFTs of a frame */
};

SpeechParams speech_defaults(void) {
    SpeechParams p = { 1.0f, 128, 1.1f, 4096 };
    return p;
}

void speech_free(Speech *s) {
    if (!s) return;
    ctx_free(s->dc);
    ctx_free(s->cc);
    model_free(&s->depth);
    model_free(&s->codec);
    free(s->hist);
    free(s->hp); free(s->x); free(s->y); free(s->logits); free(s->pos); free(s->spec); free(s->frames);
    free(s);
}

Speech *speech_load(const char *mmproj, ThreadPool *pool, int lm_dim, const char *model_name, int max_frames,
                    char *err, int errlen) {
    Speech *s = (Speech *)calloc(1, sizeof(Speech));
    if (!model_load_lfm_part(&s->depth, mmproj, "a.gen.code.blk.", "a.gen.code.output_norm.weight", 0, 1, err, errlen)) {
        free(s);
        return NULL;
    }
    if (!model_load_lfm_part(&s->codec, mmproj, "a.gen.wav.tfm.blk.", "a.gen.wav.tfm.output_norm.weight", WIN, 0, err, errlen)) {
        model_free(&s->depth);
        free(s);
        return NULL;
    }
    uint64_t bl = 0;
    const char *bn = gguf_get_str(&s->depth.gguf, "general.basename", &bl);
    if (model_name && *model_name && bn && (bl != strlen(model_name) || memcmp(bn, model_name, bl))) {
        snprintf(err, errlen, "%s is the projector of %.*s, not of %s", mmproj, (int)bl, bn, model_name);
        speech_free(s);
        return NULL;
    }
    Model *d = &s->depth, *c = &s->codec;
    s->lm_dim = lm_dim;
    s->D = d->dim;
    s->CD = c->dim;
    int ok = model_mat(d, "a.gen.code.proj_in.weight", &s->proj_in, err, errlen) &&
             (s->proj_in_b = model_f32(d, "a.gen.code.proj_in.bias", NCB * s->D, err, errlen)) != NULL &&
             (s->mask = model_f32(d, "a.gen.code.mask_embd", s->D, err, errlen)) != NULL &&
             model_mat(d, "a.gen.code.embd.weight", &s->embd, err, errlen) &&
             (s->head_norm = model_f32(d, "a.gen.code.head_norm.weight", NCB * s->D, err, errlen)) != NULL &&
             model_mat(d, "a.gen.code.out_embd.weight", &s->out_embd, err, errlen) &&
             (s->fsq = model_f32(c, "a.gen.wav.quant.first.codebook.weight", NLVL * (NCODE - 1), err, errlen)) != NULL &&
             (s->po_w = model_f32(c, "a.gen.wav.quant.first.out_proj.weight", NLVL * NCB * 256, err, errlen)) != NULL &&
             (s->po_b = model_f32(c, "a.gen.wav.quant.first.out_proj.bias", 256, err, errlen)) != NULL &&
             model_mat(c, "a.gen.wav.tfm.in_proj.weight", &s->in_proj, err, errlen) &&
             (s->in_proj_b = model_f32(c, "a.gen.wav.tfm.in_proj.bias", s->CD, err, errlen)) != NULL &&
             model_mat(c, "a.gen.wav.tfm.out_proj.weight", &s->out_proj, err, errlen) &&
             (s->out_proj_b = model_f32(c, "a.gen.wav.tfm.out_proj.bias", 2 * NBIN, err, errlen)) != NULL;
    if (ok && (s->D > 1024 || s->CD > 1024 || lm_dim > 4096 ||   /* the sizes of the buffers below */
               s->proj_in.cols != lm_dim || s->proj_in.rows != NCB * s->D || s->embd.cols != s->D ||
               s->out_embd.cols != lm_dim || s->out_embd.rows != NCB * NCODE || s->in_proj.cols != 256 ||
               s->in_proj.rows != s->CD || s->out_proj.cols != s->CD || s->out_proj.rows != 2 * NBIN)) {
        snprintf(err, errlen, "the speech decoder of %s does not fit this model", mmproj);
        ok = 0;
    }
    if (!ok) { speech_free(s); return NULL; }
    s->max_pos = (max_frames > 0 ? max_frames : 4096) * FSTACK;
    s->dc = ctx_create(d, pool, NCB, NCB);
    s->cc = ctx_create(c, pool, s->max_pos, FSTACK);
    s->hist = (int *)malloc(sizeof(int) * (size_t)(max_frames > 0 ? max_frames : 4096) * NCB);
    s->hp = (float *)malloc(sizeof(float) * NCB * s->D);
    s->x = (float *)malloc(sizeof(float) * NCB * (s->D > s->CD ? s->D : s->CD));
    s->y = (float *)malloc(sizeof(float) * NCB * (s->D > s->CD ? s->D : s->CD));
    s->logits = (float *)malloc(sizeof(float) * NCB * NCODE);
    s->pos = (float *)malloc(sizeof(float) * FSTACK * 2 * NBIN);
    s->spec = (float *)malloc(sizeof(float) * 2 * NBIN);
    s->frames = (float *)malloc(sizeof(float) * FSTACK * NFFT);
    s->pool = pool;
    if (!s->dc || !s->cc) { snprintf(err, errlen, "out of memory for the speech decoder"); speech_free(s); return NULL; }
    for (int i = 0; i < NFFT; i++) {
        s->win[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / NFFT));   /* periodic */
        s->cosv[i] = (float)cos(2 * M_PI * i / NFFT);
        s->sinv[i] = (float)sin(2 * M_PI * i / NFFT);
    }
    speech_reset(s);
    return s;
}

void speech_reset(Speech *s) {
    ctx_reset(s->cc);
    s->n_hist = 0;
    memset(s->ola, 0, sizeof(s->ola));
    memset(s->env, 0, sizeof(s->env));
    s->skip = TRIM;
}

/* ------------------------------------------------------------------ depth decoder */

/* PITHOS_SPEECH_DUMP=PREFIX: every frame's hidden state, codes and float audio, appended to
 * PREFIX.hid ([lm_dim] f32), PREFIX.codes ([12] i32), PREFIX.logits (the [2017] f32 of each
 * codebook as its pass scores it, in the order they are revealed), PREFIX.polar (the codec's
 * [1282] log magnitude and phase per position) and PREFIX.pcm (f32) */
static void dump(const char *ext, const void *p, size_t size) {
    const char *prefix = getenv("PITHOS_SPEECH_DUMP");
    if (!prefix) return;
    char path[512];
    snprintf(path, sizeof(path), "%s.%s", prefix, ext);
    FILE *f = file_open(path, "ab");
    if (f) { fwrite(p, 1, size, f); fclose(f); }
}

static uint64_t rnd(uint64_t *r) {   /* xorshift64* */
    uint64_t x = *r;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *r = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static void rms(float *o, const float *x, const float *w, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float k = (float)(1.0 / sqrt(ss / n + 1e-5));
    for (int i = 0; i < n; i++) o[i] = x[i] * k * w[i];
}

/* one pass: logits[i][NCODE] of the codebooks in want[0..nw), given the codes known so far (-1: masked) */
static void depth_pass(Speech *s, const int *codes, const int *want, int nw) {
    const int D = s->D;
    for (int i = 0; i < NCB; i++) {
        float *xi = s->x + (size_t)i * D;
        const float *hi = s->hp + (size_t)i * D;
        if (codes[i] >= 0) {
            dequant_row(s->embd.type, s->embd.data + (size_t)(i * NCODE + codes[i]) * s->embd.row_bytes, xi, D);
            for (int k = 0; k < D; k++) xi[k] += hi[k];
        } else {
            for (int k = 0; k < D; k++) xi[k] = hi[k] + s->mask[k];
        }
    }
    ctx_reset(s->dc);
    ctx_forward_embd(s->dc, s->x, NCB, s->y);
    for (int j = 0; j < nw; j++) {
        const int i = want[j];
        float v[1024];
        rms(v, s->y + (size_t)i * D, s->head_norm + (size_t)i * D, D);
        Mat sub = s->embd;
        sub.data += (size_t)i * NCODE * s->embd.row_bytes;
        sub.rows = NCODE;
        ctx_matmul(s->dc, &sub, v, 1, s->logits + (size_t)i * NCODE);
        dump("logits", s->logits + (size_t)i * NCODE, sizeof(float) * NCODE);
    }
}

static int cmp_desc(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

/* a code of codebook cb from its logits: repetition penalty over its recent codes, then
 * temperature and top-k (codes scoring at least the k-th best, ties included) */
static int sample_code(Speech *s, float *l, int cb, const SpeechParams *p, uint64_t *rng) {
    if (p->repeat_penalty != 1.f && p->repeat_last_n > 0) {
        static unsigned char seen[NCODE];
        memset(seen, 0, sizeof(seen));
        const int f0 = s->n_hist > p->repeat_last_n ? s->n_hist - p->repeat_last_n : 0;
        for (int f = f0; f < s->n_hist; f++) {
            const int c = s->hist[(size_t)f * NCB + cb];
            if (c < 0 || c >= NCODE || seen[c]) continue;
            seen[c] = 1;
            l[c] = l[c] > 0 ? l[c] / p->repeat_penalty : l[c] * p->repeat_penalty;
        }
    }
    int best = 0;
    for (int v = 1; v < NCODE; v++)
        if (l[v] > l[best]) best = v;
    if (p->temp <= 0) return best;
    float lmin = -INFINITY;
    if (p->top_k > 0 && p->top_k < NCODE) {
        float tmp[NCODE];
        memcpy(tmp, l, sizeof(tmp));
        qsort(tmp, NCODE, sizeof(float), cmp_desc);
        lmin = tmp[p->top_k - 1];
    }
    double prob[NCODE], sum = 0;
    for (int v = 0; v < NCODE; v++) {
        prob[v] = l[v] < lmin ? 0.0 : exp((double)(l[v] - l[best]) / p->temp);
        sum += prob[v];
    }
    double r = (double)(rnd(rng) >> 11) * (1.0 / 9007199254740992.0) * sum;
    for (int v = 0; v < NCODE; v++) {
        r -= prob[v];
        if (r <= 0) return v;
    }
    return best;
}

int speech_frame(Speech *s, const float *hidden, const SpeechParams *p, uint64_t *rng, int codes[NCB]) {
    dump("hid", hidden, sizeof(float) * (size_t)s->lm_dim);
    ctx_matmul(s->dc, &s->proj_in, hidden, 1, s->hp);
    for (int i = 0; i < NCB * s->D; i++) s->hp[i] += s->proj_in_b[i];
    int order[NCB];
    for (int i = 0; i < NCB; i++) order[i] = i;
    const char *fixed = getenv("PITHOS_SPEECH_ORDER");
    if (!fixed || strcmp(fixed, "fixed"))
        for (int i = NCB - 1; i > 0; i--) {   /* Fisher-Yates */
            const int j = (int)(rnd(rng) % (uint64_t)(i + 1));
            const int t = order[i]; order[i] = order[j]; order[j] = t;
        }
    for (int i = 0; i < NCB; i++) codes[i] = -1;
    static const int schedule[3] = { 4, 4, 4 };
    for (int pass = 0, k = 0; pass < 3; pass++) {
        depth_pass(s, codes, order + k, schedule[pass]);
        for (int j = 0; j < schedule[pass]; j++, k++) {
            const int i = order[k];
            codes[i] = sample_code(s, s->logits + (size_t)i * NCODE, i, p, rng);
        }
    }
    dump("codes", codes, sizeof(int) * NCB);
    int eos = 0;
    for (int i = 0; i < NCB; i++) eos |= codes[i] == EOS_CODE;
    if (eos) {
        for (int i = 0; i < NCB; i++) codes[i] = EOS_CODE;
        return 1;
    }
    if ((s->n_hist + 1) * FSTACK <= s->max_pos) {
        memcpy(s->hist + (size_t)s->n_hist * NCB, codes, sizeof(int) * NCB);
        s->n_hist++;
    }
    return 0;
}

void speech_embed(Speech *s, const int codes[NCB], float *row) {
    float tmp[4096];
    memset(row, 0, sizeof(float) * (size_t)s->lm_dim);
    for (int i = 0; i < NCB; i++) {
        const int c = codes[i] >= 0 && codes[i] < NCODE ? codes[i] : EOS_CODE;
        dequant_row(s->out_embd.type, s->out_embd.data + (size_t)(i * NCODE + c) * s->out_embd.row_bytes, tmp, s->lm_dim);
        for (int k = 0; k < s->lm_dim; k++) row[k] += tmp[k];
    }
}

/* ------------------------------------------------------------------ codec */

/* the inverse FFT of one STFT frame (log magnitude, phase), a task per codec position */
static void irfft_task(void *ctx, int t, int tid) {
    (void)tid;
    Speech *s = (Speech *)ctx;
    const float *lp = s->pos + (size_t)t * 2 * NBIN;
    float *fr = s->frames + (size_t)t * NFFT;
    float re[NBIN], im[NBIN];
    for (int k = 0; k < NBIN; k++) {
        const float mag = expf(lp[k]);
        re[k] = mag * cosf(lp[NBIN + k]);
        im[k] = mag * sinf(lp[NBIN + k]);
    }
    for (int n = 0; n < NFFT; n++) {   /* irfft, norm "backward" (1/N); the imaginary parts of DC and Nyquist do not count */
        double acc = re[0] + ((n & 1) ? -re[NBIN - 1] : re[NBIN - 1]);
        for (int k = 1, ph = n; k < NBIN - 1; k++, ph += n) {
            if (ph >= NFFT) ph %= NFFT;
            acc += 2.0 * (re[k] * s->cosv[ph] - im[k] * s->sinv[ph]);
        }
        fr[n] = (float)(acc / NFFT);
    }
}

/* one inverse FFT into the overlap-add; the HOPS samples it completes go to pcm, past the first
 * TRIM of the utterance */
static int istft_frame(Speech *s, const float *fr, float *pcm) {
    for (int n = 0; n < NFFT; n++) {
        s->ola[n] += fr[n] * s->win[n];
        s->env[n] += s->win[n] * s->win[n];
    }
    int out = 0;
    for (int n = 0; n < HOPS; n++) {
        if (s->skip > 0) { s->skip--; continue; }
        pcm[out++] = s->env[n] > 1e-11f ? s->ola[n] / s->env[n] : 0.f;
    }
    memmove(s->ola, s->ola + HOPS, sizeof(float) * (NFFT - HOPS));
    memmove(s->env, s->env + HOPS, sizeof(float) * (NFFT - HOPS));
    memset(s->ola + NFFT - HOPS, 0, sizeof(float) * HOPS);
    memset(s->env + NFFT - HOPS, 0, sizeof(float) * HOPS);
    return out;
}

int speech_decode(Speech *s, const int codes[NCB], float *pcm) {
    if (ctx_pos(s->cc) + FSTACK > s->max_pos) return 0;   /* longer than speech_load's max_frames */
    float q[NLVL * NCB], e[256], e2[1024];
    for (int i = 0; i < NCB; i++) {
        const int c = codes[i] >= 0 && codes[i] < NCODE - 1 ? codes[i] : 0;
        memcpy(q + i * NLVL, s->fsq + (size_t)c * NLVL, sizeof(float) * NLVL);
    }
    for (int r = 0; r < 256; r++) {
        double acc = s->po_b[r];
        for (int k = 0; k < NLVL * NCB; k++) acc += s->po_w[(size_t)r * NLVL * NCB + k] * q[k];
        e[r] = (float)acc;
    }
    ctx_matmul(s->cc, &s->in_proj, e, 1, e2);
    for (int i = 0; i < s->CD; i++) e2[i] += s->in_proj_b[i];
    for (int t = 0; t < FSTACK; t++) memcpy(s->x + (size_t)t * s->CD, e2, sizeof(float) * (size_t)s->CD);
    ctx_forward_embd(s->cc, s->x, FSTACK, s->y);
    ctx_matmul(s->cc, &s->out_proj, s->y, FSTACK, s->pos);
    for (int t = 0; t < FSTACK; t++) {
        float *lp = s->pos + (size_t)t * 2 * NBIN;
        for (int k = 0; k < 2 * NBIN; k++) lp[k] += s->out_proj_b[k];
        dump("polar", lp, sizeof(float) * 2 * NBIN);
    }
    pool_run(s->pool, irfft_task, s, FSTACK);
    int n = 0;
    for (int t = 0; t < FSTACK; t++) n += istft_frame(s, s->frames + (size_t)t * NFFT, pcm + n);
    dump("pcm", pcm, sizeof(float) * (size_t)n);
    return n;
}

int speech_flush(Speech *s, float *pcm) {
    /* the samples the last frame reached, without the trimmed end */
    int n = 0;
    for (int i = 0; i < NFFT - HOPS - TRIM; i++) {
        if (s->skip > 0) { s->skip--; continue; }
        pcm[n++] = s->env[i] > 1e-11f ? s->ola[i] / s->env[i] : 0.f;
    }
    dump("pcm", pcm, sizeof(float) * (size_t)n);
    speech_reset(s);
    return n;
}

size_t speech_wav_bytes(const float *pcm, int n, int rate, unsigned char *out) {
    unsigned char *h = out;
    const uint32_t data = (uint32_t)n * 2;
    const uint32_t v[] = { 36 + data, 16, 0, (uint32_t)rate, (uint32_t)rate * 2, 0, data };
    memcpy(h, "RIFF", 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    memcpy(h + 36, "data", 4);
#define W32(o, x) (h[o] = (unsigned char)(x), h[o + 1] = (unsigned char)((x) >> 8), h[o + 2] = (unsigned char)((x) >> 16), h[o + 3] = (unsigned char)((x) >> 24))
#define W16(o, x) (h[o] = (unsigned char)(x), h[o + 1] = (unsigned char)((x) >> 8))
    W32(4, v[0]); W32(16, v[1]); W16(20, 1); W16(22, 1); W32(24, v[3]); W32(28, v[4]); W16(32, 2); W16(34, 16); W32(40, v[6]);
#undef W32
#undef W16
    for (int i = 0; i < n; i++) {
        float x = pcm[i] * 32767.f;
        if (x > 32767.f) x = 32767.f;
        if (x < -32768.f) x = -32768.f;
        const short sv = (short)lrintf(x);
        out[44 + 2 * i] = (unsigned char)sv;
        out[45 + 2 * i] = (unsigned char)((unsigned short)sv >> 8);
    }
    return 44 + 2 * (size_t)n;
}

int speech_write_wav(const char *path, const float *pcm, int n, int rate) {
    FILE *f = file_open(path, "wb");
    if (!f) return 0;
    unsigned char *b = (unsigned char *)malloc(44 + 2 * (size_t)n);
    const size_t len = speech_wav_bytes(pcm, n, rate, b);
    const int ok = fwrite(b, 1, len, f) == len;
    free(b);
    return fclose(f) == 0 && ok;
}

/* the language names of the model's "tts" chat template */
static const char *speech_language(const char *code) {
    static const char *const map[][2] = {
        { "ar", "Arabic" }, { "da", "Danish" }, { "de", "German" }, { "el", "Greek" }, { "en", "American English" },
        { "en-us", "American English" }, { "en-gb", "British English" }, { "fi", "Finnish" }, { "fr", "French" },
        { "he", "Hebrew" }, { "it", "Italian" }, { "ja", "Japanese" }, { "ko", "Korean" }, { "nl", "Dutch" },
        { "pl", "Polish" }, { "pt", "Portuguese" }, { "es", "Spanish" }, { "sv", "Swedish" }, { "tr", "Turkish" },
        { "zh", "Chinese" },
    };
    char low[16];
    const size_t n = strlen(code);
    if (n >= sizeof(low)) return code;
    for (size_t i = 0; i <= n; i++) low[i] = (char)(code[i] >= 'A' && code[i] <= 'Z' ? code[i] + 32 : code[i]);
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
        if (!strcmp(low, map[i][0])) return map[i][1];
    if (n > 2 && low[2] == '-') {   /* it-IT: the language part */
        low[2] = 0;
        for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
            if (!strcmp(low, map[i][0])) return map[i][1];
    }
    return code;
}

char *speech_prompt(const char *text, const char *language, const char *voice) {
    static const char *const fmt = "<|startoftext|><|im_start|>system\nPerform TTS in %s using a %s voice.<|im_end|>\n"
                                   "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<|audio_start|>";
    const char *lang = speech_language(language && *language ? language : "en");
    const char *v = voice && !strcmp(voice, "male") ? "male" : "female";
    const size_t len = strlen(fmt) + strlen(lang) + strlen(v) + strlen(text) + 1;
    char *p = (char *)malloc(len);
    snprintf(p, len, fmt, lang, v, text);
    return p;
}
