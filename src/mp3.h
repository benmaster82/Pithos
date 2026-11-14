/* mp3.h - MP3 output of the speech (/v1/audio/speech): the system's encoder, found at run time,
 * so that the build needs no library. Windows: Media Foundation's MP3 encoder (Windows 8 and
 * later, not in the N editions without the Media Feature Pack); elsewhere libmp3lame (the
 * libmp3lame0 / lame package), which Windows also uses when Media Foundation has none. */
#ifndef PITHOS_MP3_H
#define PITHOS_MP3_H

typedef struct Mp3 Mp3;

/* receives encoded bytes as they come */
typedef void (*Mp3Out)(void *ctx, const unsigned char *data, int len);

/* An encoder of mono samples at rate Hz (24000: the speech) at kbps kilobits per second;
 * NULL with err when the system has no MP3 encoder. */
Mp3 *mp3_open(int rate, int kbps, Mp3Out out, void *ctx, char *err, int errlen);
/* n samples in [-1, 1]; the frames that are complete go to out. Returns 0 on an encoder error. */
int mp3_encode(Mp3 *e, const float *pcm, int n);
/* the rest of the stream: the last frames go to out. Returns 0 on an encoder error. */
int mp3_finish(Mp3 *e);
void mp3_close(Mp3 *e);

#endif
