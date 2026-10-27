/* sampler.h - temperature / top-k / top-p / min-p / repetition penalty sampling. */
#ifndef PITHOS_SAMPLER_H
#define PITHOS_SAMPLER_H

#include <stdint.h>

typedef struct {
    float temp;          /* <= 0: greedy */
    int top_k;           /* <= 0: disabled */
    float top_p;         /* >= 1: disabled */
    float min_p;         /* <= 0: disabled */
    float repeat_penalty;/* 1: disabled */
    int repeat_last_n;
    uint64_t seed;
} SamplerParams;

typedef struct {
    SamplerParams p;
    uint64_t rng;
    int *hist; int hist_n, hist_cap;
    int n_vocab;
    int *idx; float *val;
} Sampler;

void sampler_init(Sampler *s, const SamplerParams *p, int n_vocab);
void sampler_free(Sampler *s);
void sampler_accept(Sampler *s, int token);
void sampler_reset_history(Sampler *s);
int  sampler_sample(Sampler *s, float *logits);

#endif
