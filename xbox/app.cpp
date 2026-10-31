// app.cpp - UWP app that runs the Pithos server on an Xbox in Developer Mode and shows it on the TV.
//
// The engine is the normal C code built with PITHOS_UWP. A worker thread looks for a .gguf in the
// app's LocalState folder (uploaded through the Device Portal), then calls
// pithos_main("--server --host 0.0.0.0 ..."); engine output goes to LocalState\pithos.log.
// The screen is XAML built in code (no markup compiler): the Pithos mark, the server state, the
// addresses to connect to, the tail of the log, and a button that stops the server by closing
// the app.
#include <windows.h>
#include <io.h>
#include <share.h>
#include <stdio.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Networking.h>
#include <winrt/Windows.Networking.Connectivity.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.UI.h>
#include <winrt/Windows.UI.Text.h>
#include <winrt/Windows.UI.Xaml.h>
#include <winrt/Windows.UI.Xaml.Controls.h>
#include <winrt/Windows.UI.Xaml.Controls.Primitives.h>
#include <winrt/Windows.UI.Xaml.Documents.h>
#include <winrt/Windows.UI.Xaml.Markup.h>
#include <winrt/Windows.UI.Xaml.Media.h>
#include <winrt/Windows.UI.Xaml.Shapes.h>

extern "C" int pithos_main(int argc, char **argv);

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Media;

/* ------------------------------------------------------------------ server thread */

enum { WAITING, LOADING, RUNNING, STOPPED };
static std::atomic<int> g_state{ WAITING };   // RUNNING is decided by the screen, from the log
static std::wstring g_model;                  // file name, set before g_state leaves WAITING
static double g_model_gb;

static std::string utf8(const std::wstring &w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, nullptr, nullptr);
    return s;
}

static std::wstring wide(const std::string &s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
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
    // stdout and stderr both go to pithos.log, opened shared so the Device Portal and the screen
    // can read it. An app has no standard streams: freopen gives them real descriptors first,
    // then _dup2 points those descriptors at the shared file (closing the placeholders).
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
    g_model = model.substr(model.find_last_of(L'\\') + 1);
    g_model_gb = size / 1e9;
    g_state = LOADING;

    // the agent keeps its workspace next to the process; LocalState is the writable folder
    SetCurrentDirectoryW(local.c_str());

    std::vector<std::string> args = { "pithos", "--server", "--host", "0.0.0.0", "--port", "8080",
                                      "--no-gpu", "-m", utf8(model) };
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(&a[0]);
    argv.push_back(nullptr);
    int rc = pithos_main((int)args.size(), argv.data());
    fprintf(stderr, "pithos: server stopped (code %d)\n", rc);
    g_state = STOPPED;
}

/* ------------------------------------------------------------------ helpers */

// The last max bytes of a file (from the start of a line), read without blocking its writer.
static std::string read_tail(const std::wstring &path, size_t max) {
    CREATEFILE2_EXTENDED_PARAMETERS p = { sizeof(p) };
    p.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    HANDLE h = CreateFile2(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           OPEN_EXISTING, &p);
    if (h == INVALID_HANDLE_VALUE) return {};
    LARGE_INTEGER size = {}, off = {};
    GetFileSizeEx(h, &size);
    off.QuadPart = size.QuadPart > (LONGLONG)max ? size.QuadPart - (LONGLONG)max : 0;
    SetFilePointerEx(h, off, nullptr, FILE_BEGIN);
    std::string s((size_t)(size.QuadPart - off.QuadPart), '\0');
    DWORD got = 0;
    if (!s.empty()) ReadFile(h, &s[0], (DWORD)s.size(), &got, nullptr);
    CloseHandle(h);
    s.resize(got);
    if (off.QuadPart) s.erase(0, s.find('\n') == std::string::npos ? 0 : s.find('\n') + 1);
    return s;
}

// The log as shown: escape sequences removed, a lone carriage return (a progress line) keeps
// only the text after it, at most the last max_lines lines.
static std::wstring clean_log(const std::string &raw, size_t max_lines) {
    std::vector<std::string> lines(1);
    for (size_t i = 0; i < raw.size(); i++) {
        char c = raw[i];
        if (c == '\x1b') {
            if (i + 1 < raw.size() && raw[i + 1] == '[')
                for (i += 2; i < raw.size() && !(raw[i] >= '@' && raw[i] <= '~'); i++) {}
        } else if (c == '\r') {
            if (i + 1 >= raw.size() || raw[i + 1] != '\n') lines.back().clear();
        } else if (c == '\n') {
            lines.emplace_back();
        } else {
            lines.back() += c;
        }
    }
    while (!lines.empty() && lines.back().empty()) lines.pop_back();
    std::string out;
    for (size_t i = lines.size() > max_lines ? lines.size() - max_lines : 0; i < lines.size(); i++)
        out += lines[i] + (i + 1 < lines.size() ? "\n" : "");
    return wide(out);
}

// Rest of the line after the first occurrence of key, or "".
static std::string log_field(const std::string &log, const char *key) {
    size_t p = log.find(key);
    if (p == std::string::npos) return {};
    p += strlen(key);
    size_t e = log.find_first_of("\r\n", p);
    return log.substr(p, e == std::string::npos ? std::string::npos : e - p);
}

// IPv4 address of the console on the local network.
static std::wstring lan_address() {
    using namespace Windows::Networking;
    try {
        for (auto const &h : Connectivity::NetworkInformation::GetHostNames())
            if (h.Type() == HostNameType::Ipv4 && h.IPInformation()) return std::wstring(h.CanonicalName());
    } catch (...) {}
    return L"<xbox address>";
}

const uint32_t BG = 0x0e110e, PANEL = 0x171b17, LINE = 0x283028, GREEN = 0x107c10, TEXT = 0xf1f4f1,
               MUTED = 0x939d93, LOGTEXT = 0xc6cfc6, AMBER = 0xd9a520, RED = 0xd13438;

static SolidColorBrush brush(uint32_t rgb) {
    return SolidColorBrush(Windows::UI::Color{ 255, (uint8_t)(rgb >> 16), (uint8_t)(rgb >> 8), (uint8_t)rgb });
}

static TextBlock text(hstring const &s, double size, uint32_t color, bool bold = false) {
    TextBlock t;
    t.Text(s);
    t.FontSize(size);
    t.Foreground(brush(color));
    if (bold) t.FontWeight(Windows::UI::Text::FontWeights::SemiBold());
    t.TextWrapping(TextWrapping::Wrap);
    return t;
}

static Border panel(UIElement const &child) {
    Border b;
    b.Background(brush(PANEL));
    b.BorderBrush(brush(LINE));
    b.BorderThickness(Thickness{ 1, 1, 1, 1 });
    b.CornerRadius(CornerRadius{ 8, 8, 8, 8 });
    b.Padding(Thickness{ 18, 14, 18, 14 });
    b.Child(child);
    return b;
}

static GridLength star(double v = 1) { return GridLength{ v, GridUnitType::Star }; }
static GridLength fit() { return GridLength{ 0, GridUnitType::Auto }; }

static void rows(Grid const &g, std::initializer_list<GridLength> ls) {
    for (auto l : ls) { RowDefinition d; d.Height(l); g.RowDefinitions().Append(d); }
}
static void cols(Grid const &g, std::initializer_list<GridLength> ls) {
    for (auto l : ls) { ColumnDefinition d; d.Width(l); g.ColumnDefinitions().Append(d); }
}
static void place(Grid const &g, FrameworkElement const &e, int row, int col = 0) {
    Grid::SetRow(e, row);
    Grid::SetColumn(e, col);
    g.Children().Append(e);
}

// The Pithos mark (web/brand/pandora-mark.svg, copied into the package) as a XAML Path.
static Shapes::Path logo(double size) {
    std::wstring dir(Windows::ApplicationModel::Package::Current().InstalledLocation().Path());
    std::string svg = read_tail(dir + L"\\Assets\\pithos-mark.svg", 1 << 20);
    size_t a = svg.find(" d=\""), b = a == std::string::npos ? a : svg.find('"', a + 4);
    if (b == std::string::npos) return nullptr;
    std::wstring xaml = L"<Path xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation' Stretch='Uniform' Data='F0 " +
                        wide(svg.substr(a + 4, b - a - 4)) + L"'/>";
    Shapes::Path p = Markup::XamlReader::Load(xaml).as<Shapes::Path>();
    p.Fill(brush(GREEN));
    p.Width(size);
    p.Height(size);
    RotateTransform r;
    r.Angle(-9);
    p.RenderTransform(r);
    p.RenderTransformOrigin(Point{ 0.5f, 0.5f });
    return p;
}

/* ------------------------------------------------------------------ screen */

struct App : ApplicationT<App> {
    std::wstring m_local, m_ip;
    std::string m_log_raw, m_cpu, m_kernel;
    bool m_ready = false;
    int m_shown = -1;
    std::chrono::steady_clock::time_point m_since;
    DispatcherTimer m_timer{ nullptr };
    Border m_pill{ nullptr };
    Shapes::Ellipse m_dot{ nullptr };
    TextBlock m_status{ nullptr }, m_info{ nullptr }, m_log{ nullptr };
    ScrollViewer m_log_scroll{ nullptr };

    App() {
        RequestedTheme(ApplicationTheme::Dark);
        // the gamepad moves the focus between controls instead of driving a mouse pointer
        RequiresPointerMode(ApplicationRequiresPointerMode::WhenRequested);
    }

    void OnLaunched(Windows::ApplicationModel::Activation::LaunchActivatedEventArgs const &) {
        Window window = Window::Current();
        if (!window.Content()) {
            m_local = Windows::Storage::ApplicationData::Current().LocalFolder().Path();
            m_ip = lan_address();
            window.Content(build());
            std::thread(server_thread, m_local).detach();
            m_timer = DispatcherTimer();
            m_timer.Interval(std::chrono::milliseconds(500));
            m_timer.Tick([this](auto &&, auto &&) { tick(); });
            m_timer.Start();
            tick();
        }
        window.Activate();
    }

    UIElement build() {
        Grid root;
        root.Background(brush(BG));
        root.Padding(Thickness{ 12, 8, 12, 8 });
        root.RowSpacing(16);
        rows(root, { fit(), star() });

        // header: mark, name, state, stop button
        Grid head;
        head.ColumnSpacing(16);
        cols(head, { fit(), star(), fit(), fit() });
        if (auto mark = logo(58)) place(head, mark, 0, 0);
        StackPanel title;
        title.VerticalAlignment(VerticalAlignment::Center);
        TextBlock name = text(L"", 36, TEXT, true);
        Documents::Run word, dot;
        word.Text(L"pithos");
        dot.Text(L".");
        dot.Foreground(brush(GREEN));
        name.Inlines().Append(word);
        name.Inlines().Append(dot);
        title.Children().Append(name);
        title.Children().Append(text(L"OpenAI-compatible LLM server on Xbox", 14, MUTED));
        place(head, title, 0, 1);

        m_pill = Border();
        m_pill.CornerRadius(CornerRadius{ 18, 18, 18, 18 });
        m_pill.Padding(Thickness{ 14, 7, 18, 7 });
        m_pill.BorderThickness(Thickness{ 1, 1, 1, 1 });
        m_pill.Background(brush(PANEL));
        m_pill.VerticalAlignment(VerticalAlignment::Center);
        StackPanel pill;
        pill.Orientation(Orientation::Horizontal);
        pill.Spacing(9);
        m_dot = Shapes::Ellipse();
        m_dot.Width(11);
        m_dot.Height(11);
        m_dot.VerticalAlignment(VerticalAlignment::Center);
        m_status = text(L"", 15, TEXT, true);
        pill.Children().Append(m_dot);
        pill.Children().Append(m_status);
        m_pill.Child(pill);
        place(head, m_pill, 0, 2);

        Button stop;
        stop.Content(box_value(L"Stop server"));
        stop.VerticalAlignment(VerticalAlignment::Center);
        stop.Padding(Thickness{ 18, 8, 18, 8 });
        stop.Click([this](auto &&, auto &&) { confirm_stop(); });
        place(head, stop, 0, 3);
        place(root, head, 0);

        // body: how to connect (left), server log (right)
        Grid body;
        body.ColumnSpacing(16);
        cols(body, { star(2), star(3) });
        StackPanel info;
        info.Spacing(12);
        info.Children().Append(text(L"Connect", 18, TEXT, true));
        m_info = text(L"", 15, TEXT);
        m_info.LineHeight(26);
        info.Children().Append(m_info);
        place(body, panel(info), 0, 0);

        Grid log;
        log.RowSpacing(10);
        rows(log, { fit(), star() });
        place(log, text(L"Server log", 18, TEXT, true), 0);
        m_log = text(L"", 12, LOGTEXT);
        m_log.FontFamily(FontFamily(L"Consolas, Courier New"));
        m_log_scroll = ScrollViewer();
        m_log_scroll.Content(m_log);
        m_log_scroll.IsTabStop(true);   // focusable, so the gamepad can scroll it
        m_log_scroll.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
        place(log, m_log_scroll, 1);
        place(body, panel(log), 0, 1);
        place(root, body, 1);
        return root;
    }

    void tick() {
        std::string raw = read_tail(m_local + L"\\pithos.log", 32 << 10);
        if (raw != m_log_raw) {
            m_log_raw = raw;
            if (!m_ready && raw.find("OpenAI-compatible API") != std::string::npos) {
                m_ready = true;
                m_since = std::chrono::steady_clock::now();
            }
            if (m_cpu.empty()) m_cpu = log_field(raw, "CPU: ");
            if (m_kernel.empty()) m_kernel = log_field(raw, "kernel: ");
            bool at_end = m_log_scroll.VerticalOffset() >= m_log_scroll.ScrollableHeight() - 30;
            m_log.Text(clean_log(raw, 400));
            if (at_end) {
                m_log_scroll.UpdateLayout();
                m_log_scroll.ChangeView(nullptr, IReference<double>(m_log_scroll.ScrollableHeight()), nullptr);
            }
        }
        int state = g_state;
        if (state == LOADING && m_ready) state = RUNNING;
        static const wchar_t *names[] = { L"Waiting for a model", L"Loading the model", L"Server running", L"Server stopped" };
        static const uint32_t colors[] = { AMBER, AMBER, GREEN, RED };
        if (state != m_shown) {
            m_shown = state;
            m_status.Text(names[state]);
            m_dot.Fill(brush(colors[state]));
            m_pill.BorderBrush(brush(colors[state]));
        }
        show_info(state);
    }

    void show_info(int state) {
        std::wstring s;
        if (state == WAITING) {
            s = L"Upload a .gguf model into the app's LocalState folder from the Device Portal, for example:\n"
                L"python xbox/deploy.py --xbox " + m_ip + L" upload LFM3-1.7B-Q4_K_M.gguf\n\n"
                L"The server starts as soon as the upload is complete.";
        } else {
            wchar_t gb[32];
            swprintf(gb, 32, L"%.2f GB", g_model_gb);
            s = L"Web chat\n    http://" + m_ip + L":8080\n"
                L"OpenAI API\n    http://" + m_ip + L":8080/v1\n"
                L"Model\n    " + g_model + L" (" + gb + L")\n";
            if (!m_cpu.empty()) {
                std::wstring hw = wide(m_cpu);
                for (size_t p; (p = hw.find(L" | ")) != std::wstring::npos;) hw.replace(p, 3, L"\n    ");
                s += L"Hardware\n    " + hw + L"\n";
            }
            if (!m_kernel.empty()) s += L"Kernels\n    " + wide(m_kernel) + L"\n";
            if (state == RUNNING) {
                long long m = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - m_since).count();
                wchar_t up[64];
                swprintf(up, 64, L"Running for\n    %lld h %02lld min\n", m / 60, m % 60);
                s += up;
            }
        }
        if (m_info.Text() != s) m_info.Text(s);
    }

    fire_and_forget confirm_stop() {
        auto self = get_strong();
        ContentDialog d;
        d.Title(box_value(L"Stop the server?"));
        d.Content(box_value(L"Pithos closes and stops answering on the network. "
                            L"Start it again from My games & apps or from Dev Home."));
        d.PrimaryButtonText(L"Stop and close");
        d.CloseButtonText(L"Cancel");
        d.DefaultButton(ContentDialogButton::Close);
        if (co_await d.ShowAsync() == ContentDialogResult::Primary) Application::Current().Exit();
    }
};

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    Application::Start([](auto &&) { make<App>(); });
    return 0;
}
