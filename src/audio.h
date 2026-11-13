/* audio.h - speech input of LFM3-Audio: WAV files, the log-mel front end, and the speech encoder
 * and projector of an mmproj file, which turn 80 ms of speech into one embedding of the model. */
#ifndef PITHOS_AUDIO_H
#define PITHOS_AUDIO_H

#include "platform.h"

#define AUDIO_RATE 16000

typedef struct Audio Audio;

/* lm_dim: the embedding size of the language model; model_name: its general.basename (or "").
 * Both are checked against the projector. */
Audio *audio_load(const char *mmproj, ThreadPool *pool, int lm_dim, const char *model_name, char *err, int errlen);
void audio_free(Audio *a);

/* The samples of a WAV file (PCM 8/16/24/32-bit or float), mixed to mono and resampled to
 * AUDIO_RATE; NULL with err. The caller frees them. audio_decode_wav reads it from memory. */
float *audio_read_wav(const char *path, int *n, char *err, int errlen);
float *audio_decode_wav(const unsigned char *data, size_t len, int *n, char *err, int errlen);

/* how many embeddings audio_embed gives for n samples (one per 80 ms, rounded up) */
int audio_n_tokens(int n);

/* The embeddings [*n_out][lm_dim] of AUDIO_RATE mono samples (caller frees), NULL if too short.
 * PITHOS_AUDIO_DUMP=PREFIX writes, as f32: PREFIX.mel [128][frames], PREFIX.pre and PREFIX.hid
 * (encoder input and output, [tokens][768]) and PREFIX.emb. */
float *audio_embed(Audio *a, const float *pcm, int n, int *n_out);

#endif
