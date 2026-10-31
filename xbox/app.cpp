// app.cpp - UWP shell that runs the Pithos server on an Xbox in Developer Mode.
//
// The engine is the normal C code built with PITHOS_UWP; this file only provides the app
// entry point. It looks for a .gguf in the app's LocalState folder (uploaded through the
// Device Portal), then calls pithos_main("--server --host 0.0.0.0 ...") on a worker thread.
// Engine output goes to LocalState\pithos.log, readable from the Device Portal file explorer.
#include <windows.h>
#include <io.h>
#include <share.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.UI.Core.h>

extern "C" int pithos_main(int argc, char **argv);

using namespace winrt;
using namespace Windows::ApplicationModel::Core;
using namespace Windows::UI::Core;

static std::string utf8(const std::wstring &w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, nullptr, nullptr);
    return s;
}

// First *.gguf in dir and its size, or an empty string.
static std::wstring find_model(const std::wstring &dir, unsigned long long *size) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileExW((dir + L"\\*.gguf").c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, 0);
    if (h == INVALID_HANDLE_VALUE) return L"";
    std::wstring found = dir + L"\\" + fd.cFileName;
    *size = ((unsigned long long)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
    FindClose(h);
    return found;
}

static void server_thread(std::wstring local) {
    // stdout and stderr both go to pithos.log, opened shared so the Device Portal can read it.
    // An app has no standard streams: freopen gives them real descriptors first, then _dup2
    // points those descriptors at the shared file (closing the placeholders).
    FILE *f = _wfsopen((local + L"\\pithos.log").c_str(), L"w", _SH_DENYNO), *tmp = nullptr;
    if (f) {
        _wfreopen_s(&tmp, (local + L"\\stdout.tmp").c_str(), L"w", stdout);
        _wfreopen_s(&tmp, (local + L"\\stderr.tmp").c_str(), L"w", stderr);
        _dup2(_fileno(f), _fileno(stdout));
        _dup2(_fileno(f), _fileno(stderr));
        setvbuf(stdout, nullptr, _IONBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
        _wremove((local + L"\\stdout.tmp").c_str());
        _wremove((local + L"\\stderr.tmp").c_str());
    }

    // wait for the model: it can be uploaded while the app is running, so it is only used
    // once its size stops changing between two checks
    std::wstring model;
    unsigned long long size = 0, last = ~0ull;
    for (int i = 0;; i++) {
        model = find_model(local, &size);
        if (model.empty()) model = find_model(local + L"\\models", &size);
        if (!model.empty() && size == last) break;
        last = model.empty() ? ~0ull : size;
        if (i == 0) fprintf(stderr, "pithos: waiting for a .gguf in %s\n", utf8(local).c_str());
        Sleep(5000);
    }
    fprintf(stderr, "pithos: model %s (%.2f GB)\n", utf8(model).c_str(), size / 1e9);

    // the agent keeps its workspace next to the process; LocalState is the writable folder
    SetCurrentDirectoryW(local.c_str());

    std::vector<std::string> args = { "pithos", "--server", "--host", "0.0.0.0", "--port", "8080",
                                      "--no-gpu", "-m", utf8(model) };
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(&a[0]);
    argv.push_back(nullptr);
    int rc = pithos_main((int)args.size(), argv.data());
    fprintf(stderr, "pithos: server stopped (code %d)\n", rc);
}

struct App : implements<App, IFrameworkViewSource, IFrameworkView> {
    IFrameworkView CreateView() { return *this; }
    void Initialize(CoreApplicationView const &) {}
    void Load(hstring const &) {}
    void Uninitialize() {}
    void SetWindow(CoreWindow const &) {}

    void Run() {
        std::wstring local(Windows::Storage::ApplicationData::Current().LocalFolder().Path().c_str());
        std::thread(server_thread, local).detach();
        CoreWindow window = CoreWindow::GetForCurrentThread();
        window.Activate();
        window.Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessUntilQuit);
    }
};

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    CoreApplication::Run(make<App>());
    return 0;
}
