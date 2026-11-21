// voice.cpp - push-to-talk capture from the headset on the controller's 3.5 mm jack, and playback
// of the model's speech.
//
// voice_init (on its own thread) asks for the microphone, then keeps an AudioGraph running on the
// default capture device, which on the console is the headset ("Headset Microphone (... Virtual
// Audio Device)"). The graph runs all the time so that a press of the trigger loses no syllable;
// samples are only kept between voice_begin and voice_end. The input uses MediaCategory Speech:
// the console's voice processing is what the model transcribes well (the raw input came out as
// other words). The graph is 48 kHz with 8 channels and the voice in one of them (the centre):
// voice_end keeps the loudest channel as a mono 16-bit WAV, which the engine resamples itself.
#include <windows.h>
#include <MemoryBuffer.h>   // IMemoryBufferByteAccess
#include <math.h>
#include <stdio.h>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Devices.Enumeration.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Media.Audio.h>
#include <winrt/Windows.Media.Capture.h>
#include <winrt/Windows.Media.Devices.h>
#include <winrt/Windows.Media.MediaProperties.h>
#include <winrt/Windows.Media.Render.h>

namespace wf = winrt::Windows::Foundation;
namespace wa = winrt::Windows::Media::Audio;
namespace wm = winrt::Windows::Media;

HRESULT mic_request_access();   // mic_probe.cpp

static wa::AudioGraph g_graph{ nullptr };
static wa::AudioFrameOutputNode g_out{ nullptr };
static std::mutex g_mu;
static std::vector<float> g_pcm;           // interleaved, while recording
static std::atomic<bool> g_rec{ false };
static std::atomic<int> g_state{ 0 };      // 0 starting, 1 ready, -1 unavailable
static std::atomic<float> g_level{ -120 };  // dBFS of the last quantum's loudest channel
static int g_rate, g_ch;
static std::string g_status = "starting the microphone";

static void set_status(int state, const std::string &s) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_status = s;
    g_state = state;
    fprintf(stderr, "voice: %s\n", s.c_str());
}

/* ------------------------------------------------------------------ playback */

// The model's speech (24 kHz mono) goes to the default render device (the TV, or the headset when
// the console sends app audio there) through a frame input node at the graph's own rate: the
// samples are resampled on the way in and wait in a queue that the graph drains at its pace.
static wa::AudioGraph g_pgraph{ nullptr };
static wa::AudioFrameInputNode g_pin{ nullptr };
static std::mutex g_pmu;
static std::vector<float> g_queue;         // samples at g_prate still to play, from g_qpos
static size_t g_qpos;
static std::atomic<int> g_pstate{ 0 };     // 0 idle, 1 buffering, 2 playing
static std::atomic<bool> g_pend{ false };  // no more samples will come
static std::atomic<float> g_plevel{ 0 };   // RMS of the last quantum played
static int g_prate;
static double g_rpos;                      // resampling: position in the input of the next output sample
static float g_rlast;                      // the last input sample of the previous push

static void play_quantum(wa::AudioFrameInputNode const &node, wa::FrameInputNodeQuantumStartedEventArgs const &args) {
    const int n = args.RequiredSamples();
    if (n <= 0) return;
    wm::AudioFrame frame((uint32_t)n * sizeof(float));
    {
        auto buf = frame.LockBuffer(wm::AudioBufferAccessMode::Write);
        auto ref = buf.CreateReference();
        BYTE *data = nullptr;
        UINT32 cap = 0;
        ref.as<::Windows::Foundation::IMemoryBufferByteAccess>()->GetBuffer(&data, &cap);
        float *out = (float *)data;
        int got = 0;
        std::lock_guard<std::mutex> lock(g_pmu);
        if (g_pstate == 2) {
            const size_t avail = g_queue.size() - g_qpos;
            got = (int)(avail < (size_t)n ? avail : (size_t)n);
            memcpy(out, g_queue.data() + g_qpos, sizeof(float) * got);
            g_qpos += got;
            if (g_qpos > (1u << 20)) {   // drop what has been played now and then
                g_queue.erase(g_queue.begin(), g_queue.begin() + (ptrdiff_t)g_qpos);
                g_qpos = 0;
            }
            if (g_qpos == g_queue.size() && g_pend) g_pstate = 0;   // all played
        }
        for (int i = got; i < n; i++) out[i] = 0;   // nothing yet (or an underrun): silence
        double e = 0;
        for (int i = 0; i < n; i++) e += (double)out[i] * out[i];
        g_plevel = (float)sqrt(e / n);
    }
    node.AddFrame(frame);
}

static void play_init() {
    try {
        // the first render category the console accepts, on the default render device
        using wm::Render::AudioRenderCategory;
        const AudioRenderCategory cats[] = { AudioRenderCategory::GameMedia, AudioRenderCategory::Media,
                                             AudioRenderCategory::Other, AudioRenderCategory::Speech };
        const char *names[] = { "GameMedia", "Media", "Other", "Speech" };
        winrt::hstring id = wm::Devices::MediaDevice::GetDefaultAudioRenderId(wm::Devices::AudioDeviceRole::Default);
        winrt::Windows::Devices::Enumeration::DeviceInformation dev{ nullptr };
        if (!id.empty()) dev = winrt::Windows::Devices::Enumeration::DeviceInformation::CreateFromIdAsync(id).get();
        wa::AudioGraph graph{ nullptr };
        wa::AudioDeviceOutputNode out{ nullptr };
        for (int k = 0; k < 4 && !out; k++) {
            wa::AudioGraphSettings settings(cats[k]);
            if (dev) settings.PrimaryRenderDevice(dev);
            auto gr = wa::AudioGraph::CreateAsync(settings).get();
            if (gr.Status() != wa::AudioGraphCreationStatus::Success) {
                fprintf(stderr, "voice: audio output graph (%s): status %d, 0x%08x\n", names[k], (int)gr.Status(),
                        (unsigned)gr.ExtendedError().value);
                continue;
            }
            auto outr = gr.Graph().CreateDeviceOutputNodeAsync().get();
            if (outr.Status() != wa::AudioDeviceNodeCreationStatus::Success) {
                fprintf(stderr, "voice: audio output node (%s): status %d, 0x%08x\n", names[k], (int)outr.Status(),
                        (unsigned)outr.ExtendedError().value);
                gr.Graph().Close();
                continue;
            }
            graph = gr.Graph();
            out = outr.DeviceOutputNode();
        }
        if (!out) { fprintf(stderr, "voice: no audio output device\n"); return; }
        auto props = graph.EncodingProperties();
        props.ChannelCount(1);
        g_prate = (int)props.SampleRate();
        g_pin = graph.CreateFrameInputNode(props);
        g_pin.AddOutgoingConnection(out);
        g_pin.QuantumStarted(play_quantum);
        g_pgraph = graph;
        g_pgraph.Start();
        fprintf(stderr, "voice: speech out to \"%s\", %d Hz\n", dev ? winrt::to_string(dev.Name()).c_str() : "the default device",
                g_prate);
    } catch (winrt::hresult_error const &e) {
        fprintf(stderr, "voice: audio output error 0x%08x\n", (unsigned)e.code());
    }
}

bool voice_play_ready() { return g_pin != nullptr; }
float voice_play_level() { return g_plevel; }
int voice_play_state() { return g_pstate; }

// A new utterance: what was queued is dropped; playback waits for voice_play_start.
void voice_play_begin() {
    std::lock_guard<std::mutex> lock(g_pmu);
    g_queue.clear();
    g_qpos = 0;
    g_rpos = 0;
    g_rlast = 0;
    g_pend = false;
    g_pstate = 1;
}

// 16-bit samples at 24 kHz, resampled (linearly) to the graph's rate.
void voice_play_push(const int16_t *s, int n) {
    if (n <= 0 || !g_prate) return;
    const double step = 24000.0 / g_prate;
    std::lock_guard<std::mutex> lock(g_pmu);
    if (g_pstate == 0) return;   // stopped
    // input sample i is s[i] for i >= 0 and g_rlast for i == -1
    for (; g_rpos <= n - 1; g_rpos += step) {
        const int i = (int)floor(g_rpos);
        const double f = g_rpos - i;
        const float a = i < 0 ? g_rlast : s[i] / 32768.f, b = i + 1 < n ? s[i + 1] / 32768.f : a;
        g_queue.push_back((float)(a + (b - a) * f));
    }
    g_rpos -= n;
    g_rlast = s[n - 1] / 32768.f;
}

void voice_play_start() { if (g_pstate == 1) g_pstate = 2; }

// no more samples: playback ends when the queue is empty (and starts now if it had not)
void voice_play_finish() {
    std::lock_guard<std::mutex> lock(g_pmu);
    g_pend = true;
    if (g_pstate == 1) g_pstate = g_qpos < g_queue.size() ? 2 : 0;
}

void voice_play_stop() {
    std::lock_guard<std::mutex> lock(g_pmu);
    g_queue.clear();
    g_qpos = 0;
    g_pstate = 0;
}

// seconds queued and not yet played
double voice_play_buffered() {
    std::lock_guard<std::mutex> lock(g_pmu);
    return g_prate ? (g_queue.size() - g_qpos) / (double)g_prate : 0;
}

/* ------------------------------------------------------------------ hands-free */

// Without the trigger: a quantum is speech when it is louder than the room (a floor that follows
// the quietest level quickly downwards and slowly upwards) by 14 dB, and than -55 dBFS; 10 dB more
// while the console speaks, in case the headset hears the TV. 120 ms of speech start an utterance
// (with the 400 ms before it, so that its first syllable is there), 800 ms of quiet end it; it
// is kept if it holds at least 400 ms of speech, and ends anyway after 25 s.
static std::atomic<bool> g_hf{ false }, g_hf_speech{ false }, g_hf_playing{ false };
static double g_floor = -60, g_run, g_quiet, g_voiced;
static std::vector<float> g_pre, g_utt;   // the last 400 ms; the utterance
static std::vector<unsigned char> g_ready;
static double g_ready_secs;
static std::vector<unsigned char> make_wav(const float *x, size_t frames, int stride, int pick);

static void hands_free(const float *x, UINT32 frames, int ch, float db) {
    const double q = frames / (double)g_rate;   // seconds in this quantum
    if (db < g_floor) g_floor = db;
    else g_floor += (db - g_floor) * q / 5;   // up over about 5 s
    const double thr = fmax(g_floor + 14, -55) + (g_hf_playing ? 10 : 0);
    const bool loud = db > thr;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_hf_speech) {
        for (UINT32 i = 0; i < frames; i++) g_pre.push_back(x[i * g_ch + ch]);
        const size_t keep = (size_t)(0.4 * g_rate);
        if (g_pre.size() > keep) g_pre.erase(g_pre.begin(), g_pre.end() - (ptrdiff_t)keep);
        g_run = loud ? g_run + q : 0;
        if (g_run >= 0.12) {
            g_hf_speech = true;
            g_utt = g_pre;
            g_quiet = 0;
            g_voiced = g_run;
        }
        return;
    }
    for (UINT32 i = 0; i < frames; i++) g_utt.push_back(x[i * g_ch + ch]);
    if (loud) { g_quiet = 0; g_voiced += q; }
    else g_quiet += q;
    if (g_quiet >= 0.8 || g_utt.size() >= (size_t)(25 * g_rate)) {
        g_hf_speech = false;
        g_run = 0;
        g_pre.clear();
        if (g_voiced >= 0.4) {
            g_ready = make_wav(g_utt.data(), g_utt.size(), 1, 0);
            g_ready_secs = g_utt.size() / (double)g_rate;
        }
        g_utt.clear();
    }
}

void voice_hands_free(bool on) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_hf = on;
    g_hf_speech = false;
    g_run = 0;
    g_pre.clear();
    g_utt.clear();
    g_ready.clear();
}

// the console is speaking: a louder threshold
void voice_hands_free_playing(bool on) { g_hf_playing = on; }

// someone is speaking now
bool voice_hands_free_speech() { return g_hf_speech; }

// the last utterance as a WAV, once (empty: none since the last call)
std::vector<unsigned char> voice_hands_free_take(double *secs) {
    std::lock_guard<std::mutex> lock(g_mu);
    std::vector<unsigned char> w;
    w.swap(g_ready);
    *secs = w.empty() ? 0 : g_ready_secs;
    return w;
}

/* ------------------------------------------------------------------ capture */

static void capture_init();

void voice_init() {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    play_init();
    capture_init();
    // the graphs live in this thread's apartment: the thread stays for the life of the app
    for (;;) Sleep(INFINITE);
}

static void capture_init() {
    try {
        HRESULT hr = mic_request_access();
        if (hr != S_OK) {
            char m[160];
            snprintf(m, sizeof(m), hr == E_ACCESSDENIED
                         ? "no microphone access: allow it in Settings > Privacy > App privacy > Microphone"
                         : "no microphone (0x%08x)", (unsigned)hr);
            set_status(-1, m);
            return;
        }
        wa::AudioGraphSettings settings(wm::Render::AudioRenderCategory::Speech);
        auto gr = wa::AudioGraph::CreateAsync(settings).get();
        if (gr.Status() != wa::AudioGraphCreationStatus::Success) { set_status(-1, "no audio graph"); return; }
        g_graph = gr.Graph();
        // the default capture device, named: a node made without one has a null Device()
        winrt::hstring id = wm::Devices::MediaDevice::GetDefaultAudioCaptureId(wm::Devices::AudioDeviceRole::Communications);
        if (id.empty()) id = wm::Devices::MediaDevice::GetDefaultAudioCaptureId(wm::Devices::AudioDeviceRole::Default);
        if (id.empty()) {
            set_status(-1, "no headset microphone: plug a headset into the controller and restart");
            return;
        }
        auto device = winrt::Windows::Devices::Enumeration::DeviceInformation::CreateFromIdAsync(id).get();
        auto ir = g_graph.CreateDeviceInputNodeAsync(wm::Capture::MediaCategory::Speech, g_graph.EncodingProperties(), device).get();
        if (ir.Status() != wa::AudioDeviceNodeCreationStatus::Success) {
            set_status(-1, "no headset microphone: plug a headset into the controller and restart");
            return;
        }
        auto in = ir.DeviceInputNode();
        g_out = g_graph.CreateFrameOutputNode();
        in.AddOutgoingConnection(g_out);
        auto ep = g_graph.EncodingProperties();
        g_rate = (int)ep.SampleRate();
        g_ch = (int)ep.ChannelCount();
        g_graph.QuantumStarted([](wa::AudioGraph const &, wf::IInspectable const &) {
            auto frame = g_out.GetFrame();
            auto buf = frame.LockBuffer(wm::AudioBufferAccessMode::Read);
            auto ref = buf.CreateReference();
            BYTE *data = nullptr;
            UINT32 cap = 0;
            ref.as<::Windows::Foundation::IMemoryBufferByteAccess>()->GetBuffer(&data, &cap);
            const float *x = (const float *)data;
            const UINT32 n = buf.Length() / sizeof(float);
            if (!n || !g_ch) return;
            double best = 0;
            int loud = 0;
            for (int c = 0; c < g_ch; c++) {
                double s = 0;
                for (UINT32 i = c; i < n; i += g_ch) s += (double)x[i] * x[i];
                if (s / (n / g_ch) > best) { best = s / (n / g_ch); loud = c; }
            }
            g_level = (float)(10 * log10(best + 1e-12));
            if (g_rec) {
                std::lock_guard<std::mutex> lock(g_mu);
                g_pcm.insert(g_pcm.end(), x, x + n);
            }
            if (g_hf) hands_free(x, n / g_ch, loud, g_level);
        });
        g_graph.Start();
        char m[160];
        snprintf(m, sizeof(m), "microphone \"%s\", %d Hz x %d ch", winrt::to_string(device.Name()).c_str(), g_rate, g_ch);
        set_status(1, m);
    } catch (winrt::hresult_error const &e) {
        char m[160];
        snprintf(m, sizeof(m), "microphone error 0x%08x", (unsigned)e.code());
        set_status(-1, m);
    }
}

// 1 ready, 0 starting, -1 unavailable (voice_status says why)
int voice_state() { return g_state; }

std::string voice_status() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_status;
}

float voice_level() { return g_level; }

void voice_begin() {
    std::lock_guard<std::mutex> lock(g_mu);
    g_pcm.clear();
    g_rec = true;
}

// A mono 16-bit WAV of channel pick of x (frames of stride floats). The headset records quietly,
// and the engine takes a clip under -44 dBFS for silence: the peak is brought to -3 dBFS (at most
// +30 dB, so that a silent take stays silent).
static std::vector<unsigned char> make_wav(const float *x, size_t frames, int stride, int pick) {
    std::vector<unsigned char> w(44 + frames * 2);
    auto u32 = [&](size_t at, uint32_t v) { memcpy(&w[at], &v, 4); };
    auto u16 = [&](size_t at, uint16_t v) { memcpy(&w[at], &v, 2); };
    memcpy(&w[0], "RIFF", 4); u32(4, (uint32_t)(36 + frames * 2)); memcpy(&w[8], "WAVEfmt ", 8);
    u32(16, 16); u16(20, 1); u16(22, 1); u32(24, g_rate); u32(28, g_rate * 2); u16(32, 2); u16(34, 16);
    memcpy(&w[36], "data", 4); u32(40, (uint32_t)(frames * 2));
    float peak = 0;
    for (size_t i = 0; i < frames; i++) peak = fmaxf(peak, fabsf(x[i * stride + pick]));
    const float gain = peak > 0 ? fminf(0.708f / peak, 31.6f) : 1;
    for (size_t i = 0; i < frames; i++) {
        float a = x[i * stride + pick] * gain;
        a = a > 1 ? 1 : a < -1 ? -1 : a;
        int16_t q = (int16_t)lrintf(a * 32767);
        memcpy(&w[44 + i * 2], &q, 2);
    }
    return w;
}

// Stops keeping samples: the loudest channel as a mono 16-bit WAV, and its length in seconds.
std::vector<unsigned char> voice_end(double *secs) {
    std::vector<float> x;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_rec = false;
        x.swap(g_pcm);
    }
    *secs = 0;
    if (!g_ch || x.size() < (size_t)g_ch) return {};
    const size_t frames = x.size() / g_ch;
    int pick = 0;
    double best = -1;
    for (int c = 0; c < g_ch; c++) {
        double s = 0;
        for (size_t i = 0; i < frames; i++) s += (double)x[i * g_ch + c] * x[i * g_ch + c];
        if (s > best) { best = s; pick = c; }
    }
    *secs = frames / (double)g_rate;
    return make_wav(x.data(), frames, g_ch, pick);
}
