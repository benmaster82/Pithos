/* membw.c - multi-threaded memory read bandwidth ceiling (uses the engine thread pool). */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/platform.h"

typedef struct { const uint8_t *buf; size_t chunk; double sums[64 * 8]; } Ctx;

__attribute__((target("avx2"))) static void task(void *vc, int i, int tid) {
    Ctx *c = (Ctx *)vc;
    const __m256i *p = (const __m256i *)(c->buf + (size_t)i * c->chunk);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t k = 0; k < c->chunk / 32; k += 4) {
        a0 = _mm256_xor_si256(a0, _mm256_load_si256(p + k));
        a1 = _mm256_xor_si256(a1, _mm256_load_si256(p + k + 1));
        a2 = _mm256_xor_si256(a2, _mm256_load_si256(p + k + 2));
        a3 = _mm256_xor_si256(a3, _mm256_load_si256(p + k + 3));
    }
    a0 = _mm256_xor_si256(_mm256_xor_si256(a0, a1), _mm256_xor_si256(a2, a3));
    c->sums[tid * 8] += (double)_mm256_extract_epi32(a0, 0);
}

int main(int argc, char **argv) {
    size_t size = (size_t)1 << 30;
    uint8_t *buf = aligned_malloc(size, 64);
    memset(buf, 1, size);
    Ctx c = { buf, 64 * 1024, { 0 } };
    int nts[] = { 1, 2, 3, 4, 6, 8, 12 };
    for (size_t k = 0; k < sizeof(nts) / sizeof(nts[0]); k++) {
        ThreadPool *p = pool_create(nts[k]);
        double best = 0;
        for (int r = 0; r < 4; r++) {
            double t0 = time_sec();
            pool_run(p, task, &c, (int)(size / c.chunk));
            double bw = size / (time_sec() - t0) / 1e9;
            if (bw > best) best = bw;
        }
        printf("threads %2d: %.1f GB/s\n", nts[k], best);
        pool_destroy(p);
    }
    return 0;
}
