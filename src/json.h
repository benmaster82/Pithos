/* json.h - small JSON reader/writer for the HTTP API (UTF-8, RFC 8259). */
#ifndef PITHOS_JSON_H
#define PITHOS_JSON_H

#include <stdarg.h>

typedef enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct Json Json;
struct Json {
    JType t;
    char *key;          /* member name when the parent is an object */
    char *s;            /* J_STR: unescaped value; J_NUM: source text (both NUL-terminated) */
    int len;            /* byte length of s */
    double num;
    Json *child, *next; /* children of arrays / objects, in source order */
    int n;              /* number of children */
};

/* Parse text[0..len). Returns NULL and fills err on malformed input. */
Json *json_parse(const char *text, int len, char *err, int errlen);
void  json_free(Json *j);

Json       *json_get(const Json *obj, const char *key);   /* NULL if absent or not an object */
const char *json_str(const Json *j, const char *def);     /* string value or def */
double      json_num(const Json *j, double def);          /* number value or def */
int         json_bool(const Json *j, int def);

/* Growable byte buffer used to build responses. */
typedef struct { char *s; int n, cap; } Buf;
void buf_add(Buf *b, const char *s, int n);               /* n < 0: strlen */
void buf_printf(Buf *b, const char *fmt, ...);
void buf_free(Buf *b);
/* Append s[0..n) as a quoted JSON string (UTF-8 kept, control characters escaped). */
void buf_json_str(Buf *b, const char *s, int n);
/* Serialize j like Python's json.dumps (separators ", " and ": ", key order kept,
 * non-ASCII kept): the format chat templates see through `tojson`. */
void json_write(Buf *b, const Json *j);

#endif
