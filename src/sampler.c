/* sampler.c - see sampler.h. Order follows llama.cpp defaults:
 * penalties -> top-k -> top-p -> min-p -> temperature -> draw. */
#include "sampler.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_next(uint64_t *s) {
    /* xorshift64* */
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static float rng_f32(uint64_t *s) { return (float)((rng_next(s) >> 40) * (1.0 / 16777216.0)); }

void sampler_init(Sampler *s, const SamplerParams *p, int n_vocab) {
    memset(s, 0, sizeof(*s));
    s->p = *p;
    s->rng = p->seed ? p->seed : 0x9E3779B97F4A7C15ull;
    s->n_vocab = n_vocab;
    s->hist_cap = p->repeat_last_n > 0 ? p->repeat_last_n : 1;
    s->hist = (int *)calloc((size_t)s->hist_cap, sizeof(int));
    s->idx = (int *)malloc(sizeof(int) * n_vocab);
    s->val = (float *)malloc(sizeof(float) * n_vocab);
}

void sampler_free(Sampler *s) {
    free(s->hist); free(s->idx); free(s->val);
    memset(s, 0, sizeof(*s));
}

void sampler_accept(Sampler *s, int token) {
    if (s->hist_n < s->hist_cap) s->hist[s->hist_n++] = token;
    else {
        memmove(s->hist, s->hist + 1, sizeof(int) * (size_t)(s->hist_cap - 1));
        s->hist[s->hist_cap - 1] = token;
    }
}

void sampler_reset_history(Sampler *s) { s->hist_n = 0; }

/* partial selection: move the k largest (by val) to the front, sorted descending */
static void top_k_sort(int *idx, float *val, int n, int k) {
    /* min-heap of size k */
    for (int i = 0; i < k; i++) {
        int j = i;
        while (j > 0) {
            int p = (j - 1) / 2;
            if (val[p] <= val[j]) break;
            float tv = val[p]; val[p] = val[j]; val[j] = tv;
            int ti = idx[p]; idx[p] = idx[j]; idx[j] = ti;
            j = p;
        }
    }
    for (int i = k; i < n; i++) {
        if (val[i] <= val[0]) continue;
        val[0] = val[i]; idx[0] = idx[i];
        int j = 0;
        for (;;) {
            int l = 2 * j + 1, r = l + 1, m = j;
            if (l < k && val[l] < val[m]) m = l;
            if (r < k && val[r] < val[m]) m = r;
            if (m == j) break;
            float tv = val[m]; val[m] = val[j]; val[j] = tv;
            int ti = idx[m]; idx[m] = idx[j]; idx[j] = ti;
            j = m;
        }
    }
    /* heap sort the k elements descending */
    for (int end = k - 1; end > 0; end--) {
        float tv = val[0]; val[0] = val[end]; val[end] = tv;
        int ti = idx[0]; idx[0] = idx[end]; idx[end] = ti;
        int j = 0;
        for (;;) {
            int l = 2 * j + 1, r = l + 1, m = j;
            if (l < end && val[l] < val[m]) m = l;
            if (r < end && val[r] < val[m]) m = r;
            if (m == j) break;
            float t2 = val[m]; val[m] = val[j]; val[j] = t2;
            int i2 = idx[m]; idx[m] = idx[j]; idx[j] = i2;
            j = m;
        }
    }
}

int sampler_sample(Sampler *s, float *logits) {
    const int n = s->n_vocab;
    if (s->p.repeat_penalty != 1.f && s->hist_n > 0) {
        for (int i = 0; i < s->hist_n; i++) {
            int t = s->hist[i];
            int seen = 0;
            for (int j = 0; j < i; j++) if (s->hist[j] == t) { seen = 1; break; }
            if (seen || t < 0 || t >= n) continue;
            logits[t] = logits[t] > 0 ? logits[t] / s->p.repeat_penalty : logits[t] * s->p.repeat_penalty;
        }
    }
    if (s->p.temp <= 0) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    int k = s->p.top_k > 0 && s->p.top_k < n ? s->p.top_k : n;
    if (k > 4096) k = 4096; /* everything beyond is negligible after softmax for practical temps */
    for (int i = 0; i < n; i++) { s->idx[i] = i; s->val[i] = logits[i]; }
    top_k_sort(s->idx, s->val, n, k);
    /* softmax over candidates (with temperature) */
    const float mx = s->val[0];
    double sum = 0;
    for (int i = 0; i < k; i++) { s->val[i] = expf((s->val[i] - mx) / s->p.temp); sum += s->val[i]; }
    for (int i = 0; i < k; i++) s->val[i] = (float)(s->val[i] / sum);
    int m = k;
    if (s->p.top_p < 1.f) {
        double c = 0;
        for (int i = 0; i < k; i++) { c += s->val[i]; if (c >= s->p.top_p) { m = i + 1; break; } }
    }
    if (s->p.min_p > 0) {
        const float thr = s->val[0] * s->p.min_p;
        int j = 1;
        while (j < m && s->val[j] >= thr) j++;
        m = j;
    }
    double tot = 0;
    for (int i = 0; i < m; i++) tot += s->val[i];
    double r = rng_f32(&s->rng) * tot, c = 0;
    for (int i = 0; i < m; i++) { c += s->val[i]; if (r < c) return s->idx[i]; }
    return s->idx[m - 1];
}
