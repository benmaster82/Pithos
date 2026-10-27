/* tokenizer.h - GPT-2 style byte-level BPE (tokenizer.ggml.model = "gpt2"),
 * with the LFM2 / llama3 pre-tokenizer split and special-token parsing. */
#ifndef PITHOS_TOKENIZER_H
#define PITHOS_TOKENIZER_H

#include "gguf.h"

typedef struct Tokenizer Tokenizer;

Tokenizer *tok_load(const GGUF *g, char *err, int errlen);
void tok_free(Tokenizer *t);

/* Encode UTF-8 text. If parse_special, control tokens written literally
 * (e.g. "<|im_start|>") become single tokens. Returns number of tokens written
 * (may be > max_out: then only max_out were written). */
int tok_encode(const Tokenizer *t, const char *text, int len, int parse_special, int *out, int max_out);

/* Bytes of a token's text (raw UTF-8 bytes, not NUL-terminated). */
const char *tok_piece(const Tokenizer *t, int id, int *len);

int tok_n_vocab(const Tokenizer *t);
int tok_bos(const Tokenizer *t);
int tok_eos(const Tokenizer *t);
int tok_is_control(const Tokenizer *t, int id);
/* id of the token whose text equals s exactly, or -1 */
int tok_find(const Tokenizer *t, const char *s);

#endif
