/* mp3.c - MP3 encoding through the system's encoder (see mp3.h). Both take 24 kHz mono as it is
 * (MPEG-2); Media Foundation's goes up to 48 kbps there. */
#include "mp3.h"
#include "platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) && !defined(PITHOS_UWP)
#define PITHOS_MF 1
#define COBJMACROS
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mftransform.h>
#endif

struct Mp3 {
    Mp3Out out;
    void *ctx;
    int rate;
    short *s16;          /* conversion buffer */
    int s16_cap;
    unsigned char *obuf; /* lame's output */
    int obuf_cap;
    /* lame */
    void *lame;
    /* Media Foundation */
    long long t100ns;    /* time of the next input sample */
#ifdef PITHOS_MF
    IMFTransform *t;
    IMFSample *osample;
    int provides, com;
#endif
};

static short *to_s16(Mp3 *e, const float *x, int n) {
    if (n > e->s16_cap) {
        e->s16_cap = n;
        e->s16 = (short *)realloc(e->s16, sizeof(short) * (size_t)n);
    }
    for (int i = 0; i < n; i++) {
        float v = x[i] * 32767.f;
        if (v > 32767.f) v = 32767.f;
        if (v < -32768.f) v = -32768.f;
        e->s16[i] = (short)lrintf(v);
    }
    return e->s16;
}

/* ------------------------------------------------------------------ libmp3lame */

typedef void *(*lame_init_t)(void);
typedef int (*lame_set_t)(void *, int);
typedef int (*lame_init_params_t)(void *);
typedef int (*lame_encode_t)(void *, const short *, const short *, int, unsigned char *, int);
typedef int (*lame_flush_t)(void *, unsigned char *, int);
typedef int (*lame_close_t)(void *);

static struct {
    int tried;
    lame_init_t init;
    lame_set_t in_rate, out_rate, channels, mode, brate, quality;
    lame_init_params_t params;
    lame_encode_t encode;
    lame_flush_t flush;
    lame_close_t close;
} L;

static int lame_load(void) {
    if (L.tried) return L.init != NULL;
    L.tried = 1;
    static const char *const names[] = {
#if defined(_WIN32)
        "libmp3lame.dll", "libmp3lame-0.dll",
#elif defined(__APPLE__)
        "libmp3lame.0.dylib", "libmp3lame.dylib", "/opt/homebrew/lib/libmp3lame.dylib", "/usr/local/lib/libmp3lame.dylib",
#else
        "libmp3lame.so.0", "libmp3lame.so",
#endif
    };
    void *lib = NULL;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && !lib; i++) lib = lib_open(names[i]);
    if (!lib) return 0;
    L.in_rate = (lame_set_t)lib_sym(lib, "lame_set_in_samplerate");
    L.out_rate = (lame_set_t)lib_sym(lib, "lame_set_out_samplerate");
    L.channels = (lame_set_t)lib_sym(lib, "lame_set_num_channels");
    L.mode = (lame_set_t)lib_sym(lib, "lame_set_mode");
    L.brate = (lame_set_t)lib_sym(lib, "lame_set_brate");
    L.quality = (lame_set_t)lib_sym(lib, "lame_set_quality");
    L.params = (lame_init_params_t)lib_sym(lib, "lame_init_params");
    L.encode = (lame_encode_t)lib_sym(lib, "lame_encode_buffer");
    L.flush = (lame_flush_t)lib_sym(lib, "lame_encode_flush");
    L.close = (lame_close_t)lib_sym(lib, "lame_close");
    L.init = (lame_init_t)lib_sym(lib, "lame_init");
    if (!L.in_rate || !L.out_rate || !L.channels || !L.mode || !L.brate || !L.quality || !L.params || !L.encode ||
        !L.flush || !L.close)
        L.init = NULL;
    return L.init != NULL;
}

static int lame_open(Mp3 *e, int kbps) {
    if (!lame_load() || !(e->lame = L.init())) return 0;
    L.in_rate(e->lame, e->rate);
    L.out_rate(e->lame, e->rate);
    L.channels(e->lame, 1);
    L.mode(e->lame, 3);   /* MONO */
    L.brate(e->lame, kbps);
    L.quality(e->lame, 5);
    if (L.params(e->lame) < 0) {
        L.close(e->lame);
        e->lame = NULL;
        return 0;
    }
    return 1;
}

static void lame_room(Mp3 *e, int n) {
    const int need = n + n / 4 + 7200;   /* lame's worst case */
    if (need > e->obuf_cap) {
        e->obuf_cap = need;
        e->obuf = (unsigned char *)realloc(e->obuf, (size_t)need);
    }
}

/* ------------------------------------------------------------------ Media Foundation */

#ifdef PITHOS_MF
/* the GUIDs of mfapi.h / mftransform.h, here so that the build needs no mfuuid library */
static const GUID G_AUDIO = { 0x73647561, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
static const GUID G_PCM = { 0x00000001, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
static const GUID G_MP3 = { 0x00000055, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
static const GUID G_ENCODERS = { 0x91c64bd0, 0xf91e, 0x4d8c, { 0x92, 0x76, 0xdb, 0x24, 0x82, 0x79, 0xd9, 0x75 } };
static const GUID G_MAJOR = { 0x48eba18e, 0xf8c9, 0x4687, { 0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f } };
static const GUID G_SUBTYPE = { 0xf7e34c9a, 0x42e8, 0x4714, { 0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5 } };
static const GUID G_CHANNELS = { 0x37e48bf5, 0x645e, 0x4c5b, { 0x89, 0xde, 0xad, 0xa9, 0xe2, 0x9b, 0x69, 0x6a } };
static const GUID G_RATE = { 0x5faeeae7, 0x0290, 0x4c31, { 0x9e, 0x8a, 0xc5, 0x34, 0xf6, 0x8d, 0x9d, 0xba } };
static const GUID G_BYTES = { 0x1aab75c8, 0xcfef, 0x451c, { 0xab, 0x95, 0xac, 0x03, 0x4b, 0x8e, 0x17, 0x31 } };
static const GUID G_ALIGN = { 0x322de230, 0x9eeb, 0x43bd, { 0xab, 0x7a, 0xff, 0x41, 0x22, 0x51, 0x54, 0x1d } };
static const GUID G_BITS = { 0xf2deb57f, 0x40fa, 0x4764, { 0xaa, 0x33, 0xed, 0x4f, 0x2d, 0x1f, 0xf6, 0x69 } };
static const GUID G_IMFTRANSFORM = { 0xbf94c121, 0x5b05, 0x4e6f, { 0x80, 0x00, 0xba, 0x59, 0x89, 0x61, 0x41, 0x4d } };

typedef HRESULT(WINAPI *MFStartup_t)(ULONG, DWORD);
typedef HRESULT(WINAPI *MFTEnumEx_t)(GUID, UINT32, const MFT_REGISTER_TYPE_INFO *, const MFT_REGISTER_TYPE_INFO *,
                                     IMFActivate ***, UINT32 *);
typedef HRESULT(WINAPI *MFCreateMediaType_t)(IMFMediaType **);
typedef HRESULT(WINAPI *MFCreateSample_t)(IMFSample **);
typedef HRESULT(WINAPI *MFCreateMemoryBuffer_t)(DWORD, IMFMediaBuffer **);
typedef HRESULT(WINAPI *CoInitializeEx_t)(LPVOID, DWORD);
typedef void(WINAPI *CoUninitialize_t)(void);
typedef void(WINAPI *CoTaskMemFree_t)(LPVOID);

static struct {
    int tried, ok;
    MFTEnumEx_t enum_ex;
    MFCreateMediaType_t media_type;
    MFCreateSample_t sample;
    MFCreateMemoryBuffer_t buffer;
    CoInitializeEx_t co_init;
    CoUninitialize_t co_uninit;
    CoTaskMemFree_t co_free;
} M;

static int mf_load(void) {
    if (M.tried) return M.ok;
    M.tried = 1;
    void *mf = lib_open("mfplat.dll"), *ole = lib_open("ole32.dll");
    MFStartup_t startup = (MFStartup_t)lib_sym(mf, "MFStartup");
    M.enum_ex = (MFTEnumEx_t)lib_sym(mf, "MFTEnumEx");
    M.media_type = (MFCreateMediaType_t)lib_sym(mf, "MFCreateMediaType");
    M.sample = (MFCreateSample_t)lib_sym(mf, "MFCreateSample");
    M.buffer = (MFCreateMemoryBuffer_t)lib_sym(mf, "MFCreateMemoryBuffer");
    M.co_init = (CoInitializeEx_t)lib_sym(ole, "CoInitializeEx");
    M.co_uninit = (CoUninitialize_t)lib_sym(ole, "CoUninitialize");
    M.co_free = (CoTaskMemFree_t)lib_sym(ole, "CoTaskMemFree");
    M.ok = startup && M.enum_ex && M.media_type && M.sample && M.buffer && M.co_init && M.co_uninit && M.co_free &&
           SUCCEEDED(startup(MF_VERSION, MFSTARTUP_FULL));
    return M.ok;
}

static IMFMediaType *mf_pcm_type(int rate) {
    IMFMediaType *t = NULL;
    if (FAILED(M.media_type(&t))) return NULL;
    IMFMediaType_SetGUID(t, &G_MAJOR, &G_AUDIO);
    IMFMediaType_SetGUID(t, &G_SUBTYPE, &G_PCM);
    IMFMediaType_SetUINT32(t, &G_CHANNELS, 1);
    IMFMediaType_SetUINT32(t, &G_RATE, (UINT32)rate);
    IMFMediaType_SetUINT32(t, &G_BITS, 16);
    IMFMediaType_SetUINT32(t, &G_ALIGN, 2);
    IMFMediaType_SetUINT32(t, &G_BYTES, (UINT32)rate * 2);
    return t;
}

/* The encoder takes only the output types it lists (they carry its WAVEFORMATEX extras): the mono
 * one at rate with the highest bitrate up to kbps, else the lowest above. */
static IMFMediaType *mf_mp3_type(IMFTransform *t, int rate, int kbps) {
    IMFMediaType *best = NULL;
    UINT32 best_b = 0;
    const UINT32 want = (UINT32)kbps * 125;
    for (DWORD i = 0;; i++) {
        IMFMediaType *mt = NULL;
        if (FAILED(IMFTransform_GetOutputAvailableType(t, 0, i, &mt))) break;
        UINT32 r = 0, c = 0, b = 0;
        IMFMediaType_GetUINT32(mt, &G_RATE, &r);
        IMFMediaType_GetUINT32(mt, &G_CHANNELS, &c);
        IMFMediaType_GetUINT32(mt, &G_BYTES, &b);
        const int better = !best || (b <= want ? best_b > want || b > best_b : best_b > want && b < best_b);
        if (c == 1 && r == (UINT32)rate && better) {
            if (best) IMFMediaType_Release(best);
            best = mt;
            best_b = b;
        } else IMFMediaType_Release(mt);
    }
    return best;
}

static int mf_open(Mp3 *e, int kbps) {
    if (!mf_load()) return 0;
    const HRESULT co = M.co_init(NULL, COINIT_MULTITHREADED);
    e->com = co == S_OK || co == S_FALSE;   /* each of these needs its CoUninitialize */
    MFT_REGISTER_TYPE_INFO want = { G_AUDIO, G_MP3 };
    IMFActivate **acts = NULL;
    UINT32 na = 0;
    if (SUCCEEDED(M.enum_ex(G_ENCODERS, MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT | MFT_ENUM_FLAG_SORTANDFILTER, NULL,
                            &want, &acts, &na))) {
        for (UINT32 i = 0; i < na; i++) {
            if (!e->t) IMFActivate_ActivateObject(acts[i], &G_IMFTRANSFORM, (void **)&e->t);
            IMFActivate_Release(acts[i]);
        }
        M.co_free(acts);
    }
    if (!e->t) return 0;
    IMFMediaType *ot = mf_mp3_type(e->t, e->rate, kbps), *it = mf_pcm_type(e->rate);
    int ok = ot && it && SUCCEEDED(IMFTransform_SetOutputType(e->t, 0, ot, 0)) && SUCCEEDED(IMFTransform_SetInputType(e->t, 0, it, 0));
    if (ot) IMFMediaType_Release(ot);
    if (it) IMFMediaType_Release(it);
    MFT_OUTPUT_STREAM_INFO info = { 0 };
    ok = ok && SUCCEEDED(IMFTransform_GetOutputStreamInfo(e->t, 0, &info));
    if (ok) {
        e->provides = (info.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)) != 0;
        if (!e->provides) {
            IMFMediaBuffer *b = NULL;
            ok = SUCCEEDED(M.sample(&e->osample)) && SUCCEEDED(M.buffer(info.cbSize > 16384 ? info.cbSize : 16384, &b)) &&
                 SUCCEEDED(IMFSample_AddBuffer(e->osample, b));
            if (b) IMFMediaBuffer_Release(b);
        }
    }
    ok = ok && SUCCEEDED(IMFTransform_ProcessMessage(e->t, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)) &&
         SUCCEEDED(IMFTransform_ProcessMessage(e->t, MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0));
    if (!ok) {
        if (e->osample) IMFSample_Release(e->osample);
        IMFTransform_Release(e->t);
        e->osample = NULL;
        e->t = NULL;
        return 0;
    }
    return 1;
}

/* everything the encoder has ready goes to out */
static int mf_pull(Mp3 *e) {
    for (;;) {
        MFT_OUTPUT_DATA_BUFFER ob;
        memset(&ob, 0, sizeof(ob));
        if (!e->provides) {
            IMFMediaBuffer *b = NULL;
            if (SUCCEEDED(IMFSample_GetBufferByIndex(e->osample, 0, &b))) {
                IMFMediaBuffer_SetCurrentLength(b, 0);
                IMFMediaBuffer_Release(b);
            }
            ob.pSample = e->osample;
        }
        DWORD st = 0;
        const HRESULT hr = IMFTransform_ProcessOutput(e->t, 0, 1, &ob, &st);
        if (ob.pEvents) IMFCollection_Release(ob.pEvents);
        if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return 1;
        if (FAILED(hr)) return 0;
        IMFMediaBuffer *b = NULL;
        if (ob.pSample && SUCCEEDED(IMFSample_ConvertToContiguousBuffer(ob.pSample, &b))) {
            BYTE *p = NULL;
            DWORD len = 0;
            if (SUCCEEDED(IMFMediaBuffer_Lock(b, &p, NULL, &len))) {
                if (len) e->out(e->ctx, p, (int)len);
                IMFMediaBuffer_Unlock(b);
            }
            IMFMediaBuffer_Release(b);
        }
        if (e->provides && ob.pSample) IMFSample_Release(ob.pSample);
    }
}

static int mf_push(Mp3 *e, const float *x, int n) {
    if (n <= 0) return 1;
    IMFSample *s = NULL;
    IMFMediaBuffer *b = NULL;
    BYTE *p = NULL;
    int ok = SUCCEEDED(M.sample(&s)) && SUCCEEDED(M.buffer((DWORD)n * 2, &b)) && SUCCEEDED(IMFMediaBuffer_Lock(b, &p, NULL, NULL));
    if (ok) {
        memcpy(p, to_s16(e, x, n), (size_t)n * 2);
        IMFMediaBuffer_Unlock(b);
        IMFMediaBuffer_SetCurrentLength(b, (DWORD)n * 2);
        const long long dur = (long long)n * 10000000 / e->rate;
        ok = SUCCEEDED(IMFSample_AddBuffer(s, b)) && SUCCEEDED(IMFSample_SetSampleTime(s, e->t100ns)) &&
             SUCCEEDED(IMFSample_SetSampleDuration(s, dur));
        e->t100ns += dur;
    }
    if (ok) {
        HRESULT hr = IMFTransform_ProcessInput(e->t, 0, s, 0);
        if (hr == MF_E_NOTACCEPTING) {
            ok = mf_pull(e);
            hr = IMFTransform_ProcessInput(e->t, 0, s, 0);
        }
        ok = ok && SUCCEEDED(hr) && mf_pull(e);
    }
    if (b) IMFMediaBuffer_Release(b);
    if (s) IMFSample_Release(s);
    return ok;
}

#endif

/* ------------------------------------------------------------------ the interface */

Mp3 *mp3_open(int rate, int kbps, Mp3Out out, void *ctx, char *err, int errlen) {
    Mp3 *e = (Mp3 *)calloc(1, sizeof(Mp3));
    e->out = out;
    e->ctx = ctx;
    e->rate = rate;
#ifdef PITHOS_MF
    if (mf_open(e, kbps)) return e;
#endif
    if (lame_open(e, kbps)) return e;
#ifdef PITHOS_MF
    snprintf(err, (size_t)errlen, "no MP3 encoder: Windows' Media Foundation has none (an N edition without the Media "
                                  "Feature Pack?) and libmp3lame.dll is not installed");
#else
    snprintf(err, (size_t)errlen, "no MP3 encoder: install libmp3lame (the libmp3lame0 or lame package)");
#endif
    mp3_close(e);
    return NULL;
}

int mp3_encode(Mp3 *e, const float *pcm, int n) {
    if (n <= 0) return 1;
#ifdef PITHOS_MF
    if (e->t) return mf_push(e, pcm, n);
#endif
    lame_room(e, n);
    const int k = L.encode(e->lame, to_s16(e, pcm, n), e->s16, n, e->obuf, e->obuf_cap);
    if (k < 0) return 0;
    if (k) e->out(e->ctx, e->obuf, k);
    return 1;
}

int mp3_finish(Mp3 *e) {
#ifdef PITHOS_MF
    if (e->t) {
        return SUCCEEDED(IMFTransform_ProcessMessage(e->t, MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0)) &&
             SUCCEEDED(IMFTransform_ProcessMessage(e->t, MFT_MESSAGE_COMMAND_DRAIN, 0)) && mf_pull(e);
    }
#endif
    lame_room(e, 0);
    const int k = L.flush(e->lame, e->obuf, e->obuf_cap);
    if (k < 0) return 0;
    if (k) e->out(e->ctx, e->obuf, k);
    return 1;
}

void mp3_close(Mp3 *e) {
    if (!e) return;
#ifdef PITHOS_MF
    if (e->osample) IMFSample_Release(e->osample);
    if (e->t) IMFTransform_Release(e->t);
    if (e->com) M.co_uninit();
#endif
    if (e->lame) L.close(e->lame);
    free(e->s16);
    free(e->obuf);
    free(e);
}
