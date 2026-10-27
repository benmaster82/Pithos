/* pcache.h - prompt prefix reuse for a hybrid (recurrent + attention) model.
 *
 * The attention KV cache can be cut at any position, but the short-convolution state is
 * recurrent: rewinding needs a copy of it taken at that exact position. The cache keeps the
 * token sequence held by the context plus checkpoints of the recurrent state (taken at message
 * boundaries), so a new request only processes the part after the longest reusable prefix.
 * Finished conversations are also stored on disk (KV rows + checkpoints) and reloaded when a
 * later request — even after a restart — continues one of them. */
#ifndef PITHOS_PCACHE_H
#define PITHOS_PCACHE_H

#include "model.h"

typedef struct PCache PCache;

/* dir: folder for persisted sessions (created if missing), NULL = memory only. */
PCache *pcache_create(Context *c, const char *dir, uint64_t model_id, int max_files, uint64_t max_bytes);
void    pcache_free(PCache *pc);

/* Bring the context to the longest reusable prefix of toks[0..n): rewinds to a checkpoint or
 * loads a stored session. Unless allow_full, at least one token is left to process (the caller
 * needs logits). Returns the number of tokens reused (== ctx_pos). */
int pcache_prepare(PCache *pc, const int *toks, int n, int allow_full);
/* Process tokens after the current ones; returns the logits of the last one (NULL: context full). */
float *pcache_feed(PCache *pc, const int *toks, int n);
/* Snapshot the recurrent state at the current position (a later rewind point). */
void pcache_checkpoint(PCache *pc);
/* Write the current session to disk (no-op in memory-only mode). Returns 1 on success. */
int pcache_save(PCache *pc);
/* Forget everything (context reset). */
void pcache_clear(PCache *pc);

int pcache_n_files(const PCache *pc);

#endif
