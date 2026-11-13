/* audio.c - speech input of LFM3-Audio (see audio.h).
 *
 * Front end (as the model's feature extractor): 16 kHz, a periodic Hann window of 400 samples every
 * 160 without centering, power spectrum, 128 Slaney mel filters up to 8 kHz, log10 clamped at -12,
 * then (x + 4) / 4. The clip is right-padded so that the last frame takes its last samples.
 *
 * Encoder: three stride-2 3x3 convolutions over (mel, time), each followed by SiLU (the second and
 * third depthwise, then pointwise), so 128 mel bins x 8 frames become 256 x 16 values, then a
 * linear layer to 768; 17 LFM2 blocks (model.c, causal attention over a sliding window), a final
 * RMSNorm; projector LayerNorm, Linear, GELU (erf), Linear to the language model's size. */
#include "audio.h"
#include "model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define INV_SQRT2 0.70710678118654752440f

#define N_FFT 400
#define HOP 160
#define N_MEL 128
#define N_BIN (N_FFT / 2 + 1)
#define CH 256          /* downsampler channels */
#define TOK_BLOCK 64    /* encoder tokens per downsampler pass */

struct Audio {
    Model enc;
    ThreadPool *pool;
    int lm_dim;
    const float *w[7], *b[7];   /* downsampler a.conv1d.{0,2,3,5,6} */
    Mat pre_out, mm1, mm3;      /* mm1, mm3: dequantized to F32 (see audio_load) */
    float *mm1_f, *mm3_f;
    const float *pre_out_b, *ln_w, *ln_b, *mm1_b, *mm3_b;
    float window[N_FFT], cosv[N_FFT], sinv[N_FFT];
    float filt[N_MEL][N_BIN];
};

/* ------------------------------------------------------------------ mel filters */

static double hz_to_mel(double f) {   /* Slaney: linear to 1 kHz, logarithmic above */
    return f < 1000.0 ? 3.0 * f / 200.0 : 15.0 + log(f / 1000.0) * 27.0 / log(6.4);
}

static double mel_to_hz(double m) {
    return m < 15.0 ? 200.0 * m / 3.0 : 1000.0 * exp(log(6.4) / 27.0 * (m - 15.0));
}

/* as transformers' mel_filter_bank(norm="slaney", mel_scale="slaney") from 0 to AUDIO_RATE / 2 */
static void mel_filters(float filt[N_MEL][N_BIN]) {
    double f[N_MEL + 2];
    const double mmax = hz_to_mel(AUDIO_RATE / 2.0);
    for (int i = 0; i < N_MEL + 2; i++) f[i] = mel_to_hz(mmax * i / (N_MEL + 1));
    for (int m = 0; m < N_MEL; m++) {
        const double enorm = 2.0 / (f[m + 2] - f[m]);
        for (int k = 0; k < N_BIN; k++) {
            const double fk = (double)(AUDIO_RATE / 2) * k / (N_BIN - 1);
            const double down = (fk - f[m]) / (f[m + 1] - f[m]), up = (f[m + 2] - fk) / (f[m + 2] - f[m + 1]);
            const double v = down < up ? down : up;
            filt[m][k] = (float)(v > 0 ? v * enorm : 0);
        }
    }
}

/* ------------------------------------------------------------------ loading */

Audio *audio_load(const char *mmproj, ThreadPool *pool, int lm_dim, const char *model_name, char *err, int errlen) {
    Audio *a = (Audio *)calloc(1, sizeof(Audio));
    if (!model_load_audio_encoder(&a->enc, mmproj, err, errlen)) { free(a); return NULL; }
    /* a projector of the same size can belong to another model (LFM3-1.7B shares the vocabulary) */
    uint64_t bl = 0;
    const char *bn = gguf_get_str(&a->enc.gguf, "general.basename", &bl);
    if (model_name && *model_name && bn && (bl != strlen(model_name) || memcmp(bn, model_name, bl))) {
        snprintf(err, errlen, "%s is the projector of %.*s, not of %s", mmproj, (int)bl, bn, model_name);
        audio_free(a);
        return NULL;
    }
    a->pool = pool;
    a->lm_dim = lm_dim;
    Model *m = &a->enc;
    static const int idx[5] = { 0, 2, 3, 5, 6 };
    static const int nw[5] = { 9 * CH, 9 * CH, CH * CH, 9 * CH, CH * CH };
    char nm[64];
    int ok = 1;
    for (int i = 0; i < 5 && ok; i++) {
        snprintf(nm, sizeof(nm), "a.conv1d.%d.weight", idx[i]);
        ok = (a->w[idx[i]] = model_f32(m, nm, nw[i], err, errlen)) != NULL;
        snprintf(nm, sizeof(nm), "a.conv1d.%d.bias", idx[i]);
        ok = ok && (a->b[idx[i]] = model_f32(m, nm, CH, err, errlen)) != NULL;
    }
    ok = ok && model_mat(m, "a.pre_encode.out.weight", &a->pre_out, err, errlen) &&
         (a->pre_out_b = model_f32(m, "a.pre_encode.out.bias", m->dim, err, errlen)) != NULL &&
         (a->ln_w = model_f32(m, "mm.a.mlp.0.weight", m->dim, err, errlen)) != NULL &&
         (a->ln_b = model_f32(m, "mm.a.mlp.0.bias", m->dim, err, errlen)) != NULL &&
         model_mat(m, "mm.a.mlp.1.weight", &a->mm1, err, errlen) &&
         (a->mm1_b = model_f32(m, "mm.a.mlp.1.bias", lm_dim, err, errlen)) != NULL &&
         model_mat(m, "mm.a.mlp.3.weight", &a->mm3, err, errlen) &&
         (a->mm3_b = model_f32(m, "mm.a.mlp.3.bias", lm_dim, err, errlen)) != NULL;
    if (ok && (a->pre_out.cols != CH * N_MEL / 8 || a->pre_out.rows != m->dim || a->mm1.cols != m->dim ||
               a->mm1.rows != lm_dim || a->mm3.cols != lm_dim || a->mm3.rows != lm_dim)) {
        snprintf(err, errlen, "the audio projector does not fit this model (embedding size %d)", lm_dim);
        ok = 0;
    }
    if (!ok) { audio_free(a); return NULL; }
    /* The projector's input (a LayerNorm output) has outliers ~30 times its RMS: rounded to int8
     * like other activations, it would carry ~7% error into the embeddings (0.4% after the
     * encoder). Its two matrices in F32 (23 MB) multiply float inputs instead. */
    Mat *pm[2] = { &a->mm1, &a->mm3 };
    float **pf[2] = { &a->mm1_f, &a->mm3_f };
    for (int i = 0; i < 2; i++) {
        Mat *w = pm[i];
        float *f = (float *)malloc(sizeof(float) * (size_t)w->rows * w->cols);
        for (int r = 0; r < w->rows; r++) dequant_row(w->type, w->data + (size_t)r * w->row_bytes, f + (size_t)r * w->cols, w->cols);
        *pf[i] = f;
        w->type = GGML_F32;
        w->data = (const uint8_t *)f;
        w->row_bytes = sizeof(float) * (size_t)w->cols;
    }
    for (int i = 0; i < N_FFT; i++) {
        a->window[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / N_FFT));   /* periodic */
        a->cosv[i] = (float)cos(2 * M_PI * i / N_FFT);
        a->sinv[i] = (float)sin(2 * M_PI * i / N_FFT);
    }
    mel_filters(a->filt);
    return a;
}

void audio_free(Audio *a) {
    if (!a) return;
    model_free(&a->enc);
    free(a->mm1_f);
    free(a->mm3_f);
    free(a);
}

/* ------------------------------------------------------------------ WAV */

static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

/* the 2 * half taps of windowed-sinc resampling (Hann window) for an output sample at fraction
 * frac past input sample c0: tap k weights input c0 - half + 1 + k; normalized to sum 1 */
static void sinc_taps(double frac, int half, double cut, float *w) {
    double sum = 0, tmp[2 * 64 * 16];
    for (int k = 0; k < 2 * half; k++) {
        const double d = frac + half - 1 - k;
        double v = 0;
        if (fabs(d) < half) {
            const double xx = M_PI * d * cut;
            v = (d == 0 ? 1.0 : sin(xx) / xx) * (0.5 + 0.5 * cos(M_PI * d / half));
        }
        tmp[k] = v;
        sum += v;
    }
    for (int k = 0; k < 2 * half; k++) w[k] = (float)(sum != 0 ? tmp[k] / sum : 0);
}

static int gcd(int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; }

/* windowed-sinc resampling, 16 zero crossings each side of the narrower band. Output sample i
 * sits at input position i * rin / rout: with g = gcd(rin, rout) its fraction takes rout / g
 * values, whose taps are computed once (polyphase) unless there are too many */
static float *resample(const float *x, int n, int rin, int rout, int *n_out) {
    const long long no = (long long)n * rout / rin;
    float *y = (float *)malloc(sizeof(float) * (size_t)(no > 0 ? no : 1));
    const double cut = rout < rin ? (double)rout / rin : 1.0;
    int half = (int)ceil(16 / cut);
    if (half > 64 * 8) half = 64 * 8;   /* sinc_taps' buffer: rates up to 512 kHz */
    const int g = gcd(rin, rout), L = rout / g, M = rin / g, taps = 2 * half;
    float *tab = L <= 4096 ? (float *)malloc(sizeof(float) * (size_t)L * taps) : NULL;
    if (tab)
        for (int p = 0; p < L; p++) sinc_taps((double)p / L, half, cut, tab + (size_t)p * taps);
    float *w1 = (float *)malloc(sizeof(float) * (size_t)taps);
    for (long long i = 0; i < no; i++) {
        const long long c0 = i * M / L;
        const int p = (int)(i * M % L);
        const float *w = tab ? tab + (size_t)p * taps : w1;
        if (!tab) sinc_taps((double)p / L, half, cut, w1);
        const long long k0 = c0 - half + 1;
        double acc = 0;
        for (int k = 0; k < taps; k++)
            if (k0 + k >= 0 && k0 + k < n) acc += x[k0 + k] * w[k];
        y[i] = (float)acc;
    }
    free(tab);
    free(w1);
    *n_out = (int)no;
    return y;
}

float *audio_read_wav(const char *path, int *n_out, char *err, int errlen) {
    FILE *f = file_open(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = (uint8_t *)malloc((size_t)len + 1);
    len = (long)fread(d, 1, (size_t)len, f);
    fclose(f);
    char e2[200];
    float *pcm = audio_decode_wav(d, (size_t)len, n_out, e2, sizeof(e2));
    if (!pcm) snprintf(err, errlen, "%s: %s", path, e2);
    free(d);
    return pcm;
}

float *audio_decode_wav(const unsigned char *d, size_t size, int *n_out, char *err, int errlen) {
    const long len = (long)size;
    float *pcm = NULL;
    int fmt = 0, ch = 0, rate = 0, bits = 0;
    const uint8_t *data = NULL;
    uint32_t dlen = 0;
    if (len < 12 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4)) { snprintf(err, errlen, "not a WAV file"); return NULL; }
    for (long p = 12; p + 8 <= len;) {
        const uint32_t cl = rd32(d + p + 4);
        const uint8_t *c = d + p + 8;
        if (!memcmp(d + p, "fmt ", 4) && cl >= 16 && p + 8 + 16 <= len) {
            fmt = rd16(c);
            ch = rd16(c + 2);
            rate = (int)rd32(c + 4);
            bits = rd16(c + 14);
            if (fmt == 0xFFFE && cl >= 26) fmt = rd16(c + 24);   /* WAVE_FORMAT_EXTENSIBLE: the subformat */
        } else if (!memcmp(d + p, "data", 4)) {
            data = c;
            dlen = (uint32_t)(len - (p + 8)) < cl ? (uint32_t)(len - (p + 8)) : cl;
            break;
        }
        p += 8 + (long)cl + (cl & 1);
    }
    if (!data || ch <= 0 || rate <= 0 || !((fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32))) {
        snprintf(err, errlen, "unsupported WAV (need PCM 8/16/24/32-bit or float 32-bit)");
        return NULL;
    }
    const int bps = bits / 8, nf = (int)(dlen / ((uint32_t)bps * ch));
    pcm = (float *)malloc(sizeof(float) * (size_t)(nf > 0 ? nf : 1));
    for (int i = 0; i < nf; i++) {
        double s = 0;
        for (int k = 0; k < ch; k++) {
            const uint8_t *q = data + ((size_t)i * ch + k) * bps;
            double v;
            if (fmt == 3) { float fv; memcpy(&fv, q, 4); v = fv; }
            else if (bits == 8) v = (q[0] - 128) / 128.0;
            else if (bits == 16) v = (int16_t)rd16(q) / 32768.0;
            else if (bits == 24) v = (int32_t)((uint32_t)q[0] << 8 | (uint32_t)q[1] << 16 | (uint32_t)q[2] << 24) / 2147483648.0;
            else v = (int32_t)rd32(q) / 2147483648.0;
            s += v;
        }
        pcm[i] = (float)(s / ch);   /* channels averaged */
    }
    *n_out = nf;
    if (rate != AUDIO_RATE) {
        float *r = resample(pcm, nf, rate, AUDIO_RATE, n_out);
        free(pcm);
        pcm = r;
    }
    return pcm;
}

int audio_n_tokens(int n) {
    if (n <= 0) return 0;
    const int np = n <= N_FFT ? N_FFT : n + (HOP - (n - N_FFT) % HOP) % HOP;
    const int t0 = 1 + (np - N_FFT) / HOP, t1 = (t0 + 1) / 2, t2 = (t1 + 1) / 2;
    return (t2 + 1) / 2;
}

/* ------------------------------------------------------------------ log-mel */

typedef struct {
    const Audio *a;
    const float *pcm;
    int n, nfr;
    float *mel;
} MelArgs;

#define MEL_TASK 16   /* frames per task */

static void mel_task(void *ctx, int task, int tid) {
    (void)tid;
    const MelArgs *s = (const MelArgs *)ctx;
    const Audio *a = s->a;
    float fr[N_FFT];
    double pw[N_BIN];
    for (int t = task * MEL_TASK; t < (task + 1) * MEL_TASK && t < s->nfr; t++) {
        for (int i = 0; i < N_FFT; i++) {
            const int k = t * HOP + i;
            fr[i] = (k < s->n ? s->pcm[k] : 0.f) * a->window[i];
        }
        for (int k = 0; k < N_BIN; k++) {   /* a direct DFT: 400 is not a power of two */
            double re = 0, im = 0;
            for (int i = 0, ph = 0; i < N_FFT; i++, ph += k) {
                if (ph >= N_FFT) ph -= N_FFT;
                re += fr[i] * a->cosv[ph];
                im -= fr[i] * a->sinv[ph];
            }
            pw[k] = re * re + im * im;
        }
        for (int m = 0; m < N_MEL; m++) {
            double e = 0;
            for (int k = 0; k < N_BIN; k++) e += a->filt[m][k] * pw[k];
            const double l = e > 1e-12 ? log10(e) : -12.0;
            s->mel[(size_t)m * s->nfr + t] = (float)((l + 4.0) / 4.0);
        }
    }
}

/* mel[N_MEL][nfr] of samples padded on the right to whole frames */
static float *log_mel(const Audio *a, const float *pcm, int n, int *nfr_out) {
    const int np = n <= N_FFT ? N_FFT : n + (HOP - (n - N_FFT) % HOP) % HOP;
    const int nfr = 1 + (np - N_FFT) / HOP;
    MelArgs s = { a, pcm, n, nfr, (float *)malloc(sizeof(float) * (size_t)N_MEL * nfr) };
    pool_run(a->pool, mel_task, &s, (nfr + MEL_TASK - 1) / MEL_TASK);
    *nfr_out = nfr;
    return s.mel;
}

/* ------------------------------------------------------------------ downsampler */

/* One stride-2 3x3 convolution over (frequency, time) with padding 1, then (if pw) a pointwise
 * one; bias and SiLU after the last. Columns are global time indices: in holds [i0, i1) of a
 * signal of tin columns (zero outside), out receives [o0, o1). in_ch 1: every output channel
 * reads channel 0 (the first layer), else channel c (depthwise). */
typedef struct {
    const Audio *a;
    const float *in; int in_ch, fin, i0, i1, tin;
    float *mid, *out; int fout, o0, o1;
    const float *w, *b, *pw, *pb;
} ConvArgs;

static float silu(float x) { return x / (1.f + expf(-x)); }

static void conv_task(void *ctx, int c, int tid) {
    (void)tid;
    const ConvArgs *s = (const ConvArgs *)ctx;
    const int ci = s->in_ch == 1 ? 0 : c, wi = s->i1 - s->i0, wo = s->o1 - s->o0;
    const float *in = s->in + (size_t)ci * s->fin * wi, *w = s->w + (size_t)c * 9;
    float *o = (s->pw ? s->mid : s->out) + (size_t)c * s->fout * wo;
    for (int f = 0; f < s->fout; f++)
        for (int t = s->o0; t < s->o1; t++) {
            float acc = s->b[c];
            for (int kh = 0; kh < 3; kh++) {
                const int fi = 2 * f + kh - 1;
                if (fi < 0 || fi >= s->fin) continue;
                for (int kw = 0; kw < 3; kw++) {
                    const int ti = 2 * t + kw - 1;
                    if (ti < 0 || ti >= s->tin || ti < s->i0 || ti >= s->i1) continue;
                    acc += w[kh * 3 + kw] * in[(size_t)fi * wi + (ti - s->i0)];
                }
            }
            o[(size_t)f * wo + (t - s->o0)] = s->pw ? acc : silu(acc);
        }
}

/* pointwise: out[c][f][t] = SiLU(pb[c] + sum_k pw[c][k] mid[k][f][t]), a task per output channel */
static void pw_task(void *ctx, int c, int tid) {
    (void)tid;
    const ConvArgs *s = (const ConvArgs *)ctx;
    const size_t plane = (size_t)s->fout * (s->o1 - s->o0);
    float *o = s->out + (size_t)c * plane;
    for (size_t i = 0; i < plane; i++) o[i] = s->pb[c];
    for (int k = 0; k < CH; k++) {
        const float wk = s->pw[(size_t)c * CH + k];
        const float *m = s->mid + (size_t)k * plane;
        for (size_t i = 0; i < plane; i++) o[i] += wk * m[i];
    }
    for (size_t i = 0; i < plane; i++) o[i] = silu(o[i]);
}

static void conv_stage(ConvArgs *s, ThreadPool *pool) {
    pool_run(pool, conv_task, s, CH);
    if (s->pw) pool_run(pool, pw_task, s, CH);
}

static void dump_rows(const char *prefix, const char *ext, const float *x, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "%s.%s", prefix, ext);
    FILE *f = file_open(path, "wb");
    if (f) { fwrite(x, sizeof(float), n, f); fclose(f); }
}

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

/* ------------------------------------------------------------------ embeddings */

float *audio_embed(Audio *a, const float *pcm, int n, int *n_out) {
    *n_out = 0;
    if (n <= 0) return NULL;
    Model *m = &a->enc;
    const int D = m->dim, L = a->lm_dim;
    int T0;
    float *mel = log_mel(a, pcm, n, &T0);
    const int T1 = (T0 + 1) / 2, T2 = (T1 + 1) / 2, T3 = (T2 + 1) / 2;
    const char *dump = getenv("PITHOS_AUDIO_DUMP");
    if (dump) dump_rows(dump, "mel", mel, (size_t)N_MEL * T0);
    /* downsampler, TOK_BLOCK encoder tokens at a time; then the linear layer to D */
    float *x = (float *)malloc(sizeof(float) * (size_t)T3 * D);
    float *flat = (float *)malloc(sizeof(float) * (size_t)TOK_BLOCK * CH * (N_MEL / 8));
    float *s0 = NULL, *s1 = NULL, *s2 = NULL, *s3 = NULL, *mid = NULL;
    Context *c = ctx_create(m, a->pool, T3, T3 < 512 ? T3 : 512);
    for (int b0 = 0; b0 < T3; b0 += TOK_BLOCK) {
        const int b1 = imin(b0 + TOK_BLOCK, T3);
        /* the columns each stage needs: out column o reads 2o-1 .. 2o+1 */
        const int c2a = imax(2 * b0 - 1, 0), c2b = imin(2 * b1, T2);
        const int c1a = imax(2 * c2a - 1, 0), c1b = imin(2 * c2b, T1);
        const int c0a = imax(2 * c1a - 1, 0), c0b = imin(2 * c1b, T0);
        free(s0); free(s1); free(s2); free(s3); free(mid);
        s0 = (float *)malloc(sizeof(float) * (size_t)N_MEL * (c0b - c0a));
        for (int f = 0; f < N_MEL; f++)
            memcpy(s0 + (size_t)f * (c0b - c0a), mel + (size_t)f * T0 + c0a, sizeof(float) * (size_t)(c0b - c0a));
        s1 = (float *)malloc(sizeof(float) * (size_t)CH * 64 * (c1b - c1a));
        s2 = (float *)malloc(sizeof(float) * (size_t)CH * 32 * (c2b - c2a));
        s3 = (float *)malloc(sizeof(float) * (size_t)CH * 16 * (b1 - b0));
        mid = (float *)malloc(sizeof(float) * (size_t)CH * 32 * (c2b - c2a));
        ConvArgs g1 = { a, s0, 1, N_MEL, c0a, c0b, T0, NULL, s1, 64, c1a, c1b, a->w[0], a->b[0], NULL, NULL };
        ConvArgs g2 = { a, s1, CH, 64, c1a, c1b, T1, mid, s2, 32, c2a, c2b, a->w[2], a->b[2], a->w[3], a->b[3] };
        ConvArgs g3 = { a, s2, CH, 32, c2a, c2b, T2, mid, s3, 16, b0, b1, a->w[5], a->b[5], a->w[6], a->b[6] };
        conv_stage(&g1, a->pool);
        conv_stage(&g2, a->pool);
        conv_stage(&g3, a->pool);
        const int nb = b1 - b0;
        for (int t = 0; t < nb; t++)   /* [c][f] per token, as the reference's flatten of (channel, frequency) */
            for (int ch = 0; ch < CH; ch++)
                for (int f = 0; f < 16; f++)
                    flat[(size_t)t * CH * 16 + ch * 16 + f] = s3[((size_t)ch * 16 + f) * nb + t];
        ctx_matmul(c, &a->pre_out, flat, nb, x + (size_t)b0 * D);
        for (int t = b0; t < b1; t++)
            for (int i = 0; i < D; i++) x[(size_t)t * D + i] += a->pre_out_b[i];
    }
    free(s0); free(s1); free(s2); free(s3); free(mid); free(flat); free(mel);
    /* encoder */
    float *hid = (float *)malloc(sizeof(float) * (size_t)T3 * D);
    if (dump) dump_rows(dump, "pre", x, (size_t)T3 * D);
    ctx_forward_embd(c, x, T3, hid);
    if (dump) dump_rows(dump, "hid", hid, (size_t)T3 * D);
    /* projector */
    for (int t = 0; t < T3; t++) {
        float *h = hid + (size_t)t * D;
        double mu = 0, var = 0;
        for (int i = 0; i < D; i++) mu += h[i];
        mu /= D;
        for (int i = 0; i < D; i++) var += (h[i] - mu) * (h[i] - mu);
        const float inv = (float)(1.0 / sqrt(var / D + 1e-5));
        for (int i = 0; i < D; i++) h[i] = (float)((h[i] - mu) * inv) * a->ln_w[i] + a->ln_b[i];
    }
    float *p1 = (float *)malloc(sizeof(float) * (size_t)T3 * L), *out = (float *)malloc(sizeof(float) * (size_t)T3 * L);
    ctx_matmul(c, &a->mm1, hid, T3, p1);
    for (int t = 0; t < T3; t++)
        for (int i = 0; i < L; i++) {
            const float v = p1[(size_t)t * L + i] + a->mm1_b[i];
            p1[(size_t)t * L + i] = 0.5f * v * (1.f + erff(v * INV_SQRT2));
        }
    ctx_matmul(c, &a->mm3, p1, T3, out);
    for (int t = 0; t < T3; t++)
        for (int i = 0; i < L; i++) out[(size_t)t * L + i] += a->mm3_b[i];
    if (dump) dump_rows(dump, "emb", out, (size_t)T3 * L);
    ctx_free(c);
    free(hid); free(p1); free(x);
    *n_out = T3;
    return out;
}
