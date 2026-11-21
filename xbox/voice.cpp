// voice.cpp - push-to-talk capture from the headset on the controller's 3.5 mm jack.
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

void voice_init() {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
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
        auto ir = g_graph.CreateDeviceInputNodeAsync(wm::Capture::MediaCategory::Speech).get();
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
            for (int c = 0; c < g_ch; c++) {
                double s = 0;
                for (UINT32 i = c; i < n; i += g_ch) s += (double)x[i] * x[i];
                best = fmax(best, s / (n / g_ch));
            }
            g_level = (float)(10 * log10(best + 1e-12));
            if (g_rec) {
                std::lock_guard<std::mutex> lock(g_mu);
                g_pcm.insert(g_pcm.end(), x, x + n);
            }
        });
        g_graph.Start();
        char m[160];
        snprintf(m, sizeof(m), "microphone \"%s\", %d Hz x %d ch", winrt::to_string(in.Device().Name()).c_str(), g_rate, g_ch);
        set_status(1, m);
        // the graph lives in this thread's apartment: the thread stays for the life of the app
        for (;;) Sleep(INFINITE);
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
    std::vector<unsigned char> w(44 + frames * 2);
    auto u32 = [&](size_t at, uint32_t v) { memcpy(&w[at], &v, 4); };
    auto u16 = [&](size_t at, uint16_t v) { memcpy(&w[at], &v, 2); };
    memcpy(&w[0], "RIFF", 4); u32(4, (uint32_t)(36 + frames * 2)); memcpy(&w[8], "WAVEfmt ", 8);
    u32(16, 16); u16(20, 1); u16(22, 1); u32(24, g_rate); u32(28, g_rate * 2); u16(32, 2); u16(34, 16);
    memcpy(&w[36], "data", 4); u32(40, (uint32_t)(frames * 2));
    for (size_t i = 0; i < frames; i++) {
        float a = x[i * g_ch + pick];
        a = a > 1 ? 1 : a < -1 ? -1 : a;
        int16_t q = (int16_t)lrintf(a * 32767);
        memcpy(&w[44 + i * 2], &q, 2);
    }
    return w;
}
