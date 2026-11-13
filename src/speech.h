/* speech.h - speech output of LFM3-Audio: the depth decoder (MaskGIT), which turns a hidden state
 * of the language model into a frame of 12 codes every 80 ms, and the LiquidCodec decoder, which
 * turns the frames into 24 kHz audio. */
#ifndef PITHOS_SPEECH_H
#define PITHOS_SPEECH_H

#include <stddef.h>
#include <stdint.h>
#include "platform.h"

#define SPEECH_RATE 24000
#define SPEECH_CODEBOOKS 12
#define SPEECH_FRAME 1920        /* samples per frame */

typedef struct Speech Speech;

typedef struct {
    float temp;                  /* <= 0: greedy */
    int top_k;                   /* <= 0: off */
    float repeat_penalty;        /* 1: off */
    int repeat_last_n;           /* frames */
} SpeechParams;

/* the model card's sampling: temperature 1, top-k 128, penalty 1.1 over 4096 frames */
SpeechParams speech_defaults(void);

/* lm_dim and model_name (general.basename, or "") are checked against the projector.
 * max_frames: the longest audio of one speech_reset .. speech_flush (for the codec's cache). */
Speech *speech_load(const char *mmproj, ThreadPool *pool, int lm_dim, const char *model_name, int max_frames,
                    char *err, int errlen);
void speech_free(Speech *s);

/* A new utterance: the codec state, the inverse STFT and the repetition history start over. */
void speech_reset(Speech *s);

/* The codes of the next frame from the language model's hidden state; returns 1 at the end of the
 * audio (then codes are all the end code). PITHOS_SPEECH_ORDER=fixed reveals the codebooks in
 * index order instead of a random one (for tests). */
int speech_frame(Speech *s, const float *hidden, const SpeechParams *p, uint64_t *rng, int codes[SPEECH_CODEBOOKS]);

/* The input row of the frame for the language model's next step: the sum of its codes' embeddings. */
void speech_embed(Speech *s, const int codes[SPEECH_CODEBOOKS], float *row);

/* The audio of one frame: writes up to SPEECH_FRAME samples to pcm, returns how many (the first
 * frame gives fewer: the inverse STFT holds back its overlap). speech_flush writes the rest. */
int speech_decode(Speech *s, const int codes[SPEECH_CODEBOOKS], float *pcm);
int speech_flush(Speech *s, float *pcm);

/* 16-bit mono WAV; 0 if the file cannot be written. speech_wav_bytes writes the same into
 * out (44 + 2 n bytes) and returns its size. */
int speech_write_wav(const char *path, const float *pcm, int n, int rate);
size_t speech_wav_bytes(const float *pcm, int n, int rate, unsigned char *out);

/* The model's TTS prompt (its "tts" chat template) for text, a language (ISO code such as "it",
 * or a name) and a voice ("female" or "male"); the caller frees it. */
char *speech_prompt(const char *text, const char *language, const char *voice);

#endif
