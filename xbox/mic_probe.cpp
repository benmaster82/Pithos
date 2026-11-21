// mic_probe.cpp - finds out whether this app can hear the headset plugged into a controller (the
// 3.5 mm jack): it lists the controllers' headsets and the audio capture devices, then records two
// 5-second takes through an AudioGraph, one raw (MediaCategory Other) and one with the system's
// voice processing (Speech), into LocalState\mic-raw.wav and mic-speech.wav. Runs when
// LocalState\mic-probe exists; everything it finds goes to the log.
#include <windows.h>
#include <MemoryBuffer.h>   // IMemoryBufferByteAccess
#include <stdio.h>
#include <math.h>
#include <mutex>
#include <string>
#include <vector>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Devices.Enumeration.h>
#include <winrt/Windows.Gaming.Input.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Media.Audio.h>
#include <winrt/Windows.Media.Capture.h>
#include <winrt/Windows.Media.Devices.h>
#include <winrt/Windows.Media.MediaProperties.h>
#include <winrt/Windows.Media.Render.h>

namespace wf = winrt::Windows::Foundation;
namespace wd = winrt::Windows::Devices::Enumeration;
namespace wg = winrt::Windows::Gaming::Input;
namespace wa = winrt::Windows::Media::Audio;
namespace wm = winrt::Windows::Media;

static std::string u8(winrt::hstring const &h) { return winrt::to_string(h); }

// 16-bit PCM WAV of interleaved float samples.
static bool write_wav(const std::wstring &path, const std::vector<float> &x, int rate, int channels) {
    FILE *f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    unsigned data = (unsigned)x.size() * 2, v;
    unsigned short s;
    fwrite("RIFF", 1, 4, f); v = 36 + data; fwrite(&v, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f);
    v = 16; fwrite(&v, 4, 1, f); s = 1; fwrite(&s, 2, 1, f); s = (unsigned short)channels; fwrite(&s, 2, 1, f);
    v = rate; fwrite(&v, 4, 1, f); v = rate * channels * 2; fwrite(&v, 4, 1, f);
    s = (unsigned short)(channels * 2); fwrite(&s, 2, 1, f); s = 16; fwrite(&s, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (float a : x) {
        float c = a > 1 ? 1 : a < -1 ? -1 : a;
        short q = (short)lrintf(c * 32767);
        fwrite(&q, 2, 1, f);
    }
    fclose(f);
    return true;
}

// The headset of the first controller that has one: its capture endpoint id, or "".
static winrt::hstring headset_capture() {
    // the controller list fills in shortly after the app starts
    for (int i = 0; i < 30 && wg::Gamepad::Gamepads().Size() == 0; i++) Sleep(100);
    winrt::hstring found;
    auto pads = wg::Gamepad::Gamepads();
    fprintf(stderr, "mic-probe: %u gamepad(s)\n", pads.Size());
    for (uint32_t i = 0; i < pads.Size(); i++) {
        auto h = pads.GetAt(i).Headset();
        if (!h) { fprintf(stderr, "mic-probe:   gamepad %u: no headset\n", i); continue; }
        fprintf(stderr, "mic-probe:   gamepad %u headset capture=%s render=%s\n", i,
                u8(h.CaptureDeviceId()).c_str(), u8(h.RenderDeviceId()).c_str());
        if (found.empty()) found = h.CaptureDeviceId();
    }
    return found;
}

static void list_devices() {
    using wm::Devices::MediaDevice;
    using wm::Devices::AudioDeviceRole;
    fprintf(stderr, "mic-probe: default capture %s, communications %s\n",
            u8(MediaDevice::GetDefaultAudioCaptureId(AudioDeviceRole::Default)).c_str(),
            u8(MediaDevice::GetDefaultAudioCaptureId(AudioDeviceRole::Communications)).c_str());
    fprintf(stderr, "mic-probe: default render %s\n",
            u8(MediaDevice::GetDefaultAudioRenderId(AudioDeviceRole::Default)).c_str());
    auto all = wd::DeviceInformation::FindAllAsync(MediaDevice::GetAudioCaptureSelector()).get();
    fprintf(stderr, "mic-probe: %u capture device(s)\n", all.Size());
    for (auto const &d : all)
        fprintf(stderr, "mic-probe:   \"%s\" enabled=%d id=%s\n", u8(d.Name()).c_str(), d.IsEnabled(), u8(d.Id()).c_str());
}

// The capture device to record from: the headset's endpoint if it resolves, else the system default (null).
static wd::DeviceInformation pick_device(winrt::hstring const &headset) {
    if (headset.empty()) return nullptr;
    try {
        auto d = wd::DeviceInformation::CreateFromIdAsync(headset).get();
        fprintf(stderr, "mic-probe: headset endpoint resolves to \"%s\"\n", u8(d.Name()).c_str());
        return d;
    } catch (winrt::hresult_error const &e) {
        fprintf(stderr, "mic-probe: headset endpoint does not resolve (0x%08x %s)\n", (unsigned)e.code(),
                u8(e.message()).c_str());
    }
    // the gamepad's id may name the endpoint differently: look for it among the capture devices
    auto all = wd::DeviceInformation::FindAllAsync(wm::Devices::MediaDevice::GetAudioCaptureSelector()).get();
    for (auto const &d : all) {
        std::wstring id(d.Id()), want(headset);
        if (id.find(want) != std::wstring::npos || want.find(id) != std::wstring::npos) return d;
    }
    return nullptr;
}

// Records `seconds` from `device` (null: the default) into `path`. Logs the format, the latency,
// the level and how long the first samples took to arrive.
static void record(const std::wstring &path, wd::DeviceInformation const &device, wm::Capture::MediaCategory cat,
                   const char *label, int seconds) {
    wa::AudioGraphSettings settings(wm::Render::AudioRenderCategory::Speech);
    settings.QuantumSizeSelectionMode(wa::QuantumSizeSelectionMode::LowestLatency);
    auto gr = wa::AudioGraph::CreateAsync(settings).get();
    if (gr.Status() != wa::AudioGraphCreationStatus::Success) {
        fprintf(stderr, "mic-probe: %s: AudioGraph failed (status %d, 0x%08x)\n", label, (int)gr.Status(),
                (unsigned)gr.ExtendedError().value);
        return;
    }
    auto graph = gr.Graph();
    auto ir = device ? graph.CreateDeviceInputNodeAsync(cat, graph.EncodingProperties(), device).get()
                     : graph.CreateDeviceInputNodeAsync(cat).get();
    if (ir.Status() != wa::AudioDeviceNodeCreationStatus::Success) {
        // 2 = DeviceNotAvailable, 3 = FormatNotSupported, 4 = UnknownFailure, 5 = AccessDenied (consent)
        fprintf(stderr, "mic-probe: %s: input node failed (status %d, 0x%08x)\n", label, (int)ir.Status(),
                (unsigned)ir.ExtendedError().value);
        graph.Close();
        return;
    }
    auto in = ir.DeviceInputNode();
    auto out = graph.CreateFrameOutputNode();
    in.AddOutgoingConnection(out);
    auto ep = graph.EncodingProperties();
    const int rate = (int)ep.SampleRate(), ch = (int)ep.ChannelCount();
    fprintf(stderr, "mic-probe: %s: device \"%s\", graph %s %d Hz x %d ch %d bit, quantum %d samples, latency %d samples\n",
            label, u8(in.Device().Name()).c_str(), u8(ep.Subtype()).c_str(), rate, ch, (int)ep.BitsPerSample(),
            graph.SamplesPerQuantum(), graph.LatencyInSamples());
    auto iep = in.EncodingProperties();
    fprintf(stderr, "mic-probe: %s: input node %d Hz x %d ch\n", label, (int)iep.SampleRate(), (int)iep.ChannelCount());

    std::vector<float> pcm;
    std::mutex mu;
    LARGE_INTEGER freq, t0, first = {};
    QueryPerformanceFrequency(&freq);
    auto token = graph.QuantumStarted([&](wa::AudioGraph const &, wf::IInspectable const &) {
        auto frame = out.GetFrame();
        auto buf = frame.LockBuffer(wm::AudioBufferAccessMode::Read);
        auto ref = buf.CreateReference();
        auto access = ref.as<::Windows::Foundation::IMemoryBufferByteAccess>();
        BYTE *data = nullptr;
        UINT32 cap = 0;
        access->GetBuffer(&data, &cap);
        UINT32 n = buf.Length() / sizeof(float);
        if (!n) return;
        std::lock_guard<std::mutex> lock(mu);
        if (!first.QuadPart) QueryPerformanceCounter(&first);
        pcm.insert(pcm.end(), (float *)data, (float *)data + n);
    });
    fprintf(stderr, "mic-probe: %s: SPEAK NOW for %d seconds\n", label, seconds);
    QueryPerformanceCounter(&t0);
    graph.Start();
    Sleep(seconds * 1000);
    graph.Stop();
    graph.QuantumStarted(token);
    graph.Close();

    std::lock_guard<std::mutex> lock(mu);
    double peak = 0, sum = 0;
    for (float a : pcm) { peak = fmax(peak, fabs(a)); sum += (double)a * a; }
    const double rms = pcm.empty() ? 0 : sqrt(sum / pcm.size());
    fprintf(stderr, "mic-probe: %s: %.2f s recorded, peak %.1f dBFS, rms %.1f dBFS, first samples after %.0f ms\n",
            label, ch && rate ? pcm.size() / (double)ch / rate : 0, 20 * log10(peak + 1e-9), 20 * log10(rms + 1e-9),
            first.QuadPart ? (first.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart : -1.0);
    if (!pcm.empty() && rate && ch)
        fprintf(stderr, "mic-probe: %s: %s %ls\n", label, write_wav(path, pcm, rate, ch) ? "wrote" : "could not write",
                path.c_str());
}

void mic_probe(const std::wstring &local) {
    // WinRT on this worker thread: its own multithreaded apartment (the UI thread keeps its own)
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try {
        winrt::hstring headset = headset_capture();
        list_devices();
        auto device = pick_device(headset);
        if (!device) fprintf(stderr, "mic-probe: recording from the default capture device\n");
        record(local + L"\\mic-raw.wav", device, wm::Capture::MediaCategory::Other, "raw", 5);
        Sleep(1000);
        record(local + L"\\mic-speech.wav", device, wm::Capture::MediaCategory::Speech, "speech", 5);
    } catch (winrt::hresult_error const &e) {
        fprintf(stderr, "mic-probe: error 0x%08x %s\n", (unsigned)e.code(), u8(e.message()).c_str());
    }
    fprintf(stderr, "mic-probe: done\n");
    winrt::uninit_apartment();
}
