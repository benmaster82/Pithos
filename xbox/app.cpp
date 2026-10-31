// app.cpp - UWP app that runs the Pithos server on an Xbox in Developer Mode and shows it on the TV.
//
// The engine is the normal C code built with PITHOS_UWP. A worker thread looks for a .gguf in the
// app's LocalState folder (uploaded through the Device Portal), then calls
// pithos_main("--server --host 0.0.0.0 ..."); engine output goes to LocalState\pithos.log.
// The screen is XAML built in code (no markup compiler): the Pithos mark, the server state, a
// chat with the model, the addresses to connect to, the tail of the log, and a button that stops
// the server by closing the app. The chat talks to the engine in process (server_local_chat, the
// same /v1/chat/completions handler the network uses) and sends no tools: it cannot run anything.
#include <windows.h>
#include <io.h>
#include <share.h>
#include <stdio.h>
#include <atomic>
#include <chrono>
#include <functional>
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
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.h>
#include <winrt/Windows.UI.Core.h>
#include <winrt/Windows.UI.Text.h>
#include <winrt/Windows.UI.Xaml.h>
#include <winrt/Windows.UI.Xaml.Controls.h>
#include <winrt/Windows.UI.Xaml.Controls.Primitives.h>
#include <winrt/Windows.UI.Xaml.Documents.h>
#include <winrt/Windows.UI.Xaml.Input.h>
#include <winrt/Windows.UI.Xaml.Markup.h>
#include <winrt/Windows.UI.Xaml.Media.h>
#include <winrt/Windows.UI.Xaml.Shapes.h>

extern "C" {
#include "../src/json.h"
int pithos_main(int argc, char **argv);
void server_local_chat(const char *body, int len, void (*sink)(void *u, const char *s, int n), void *u,
                       volatile int *stop);
}

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

static std::string read_tail(const std::wstring &path, size_t max);

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
    // LocalState\pithos.args: more options (for example "-t 6"), separated by spaces or lines;
    // they come after the defaults, so they also override them
    size_t defaults = args.size();
    std::string extra = read_tail(local + L"\\pithos.args", 4096);
    for (size_t p = 0; (p = extra.find_first_not_of(" \t\r\n", p)) != std::string::npos;) {
        size_t e = extra.find_first_of(" \t\r\n", p);
        args.push_back(extra.substr(p, e == std::string::npos ? e : e - p));
        p = e;
    }
    if (!extra.empty()) {
        fprintf(stderr, "pithos: options from pithos.args:");
        for (size_t i = defaults; i < args.size(); i++) fprintf(stderr, " %s", args[i].c_str());
        fprintf(stderr, "\n");
    }
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

/* ------------------------------------------------------------------ chat stream */

// Parses the response that server_local_chat hands over: the HTTP head, then SSE events
// ("data: {...}\n\n") with the reply in choices[0].delta.content / reasoning_content.
struct ChatStream {
    std::string buf;
    bool head = false, ok = true;
    std::function<void(std::string, std::string, std::string)> on_delta;   // content, reasoning, error
};

static void chat_sink(void *u, const char *s, int n) {
    ChatStream *st = (ChatStream *)u;
    st->buf.append(s, n);
    if (!st->head) {
        size_t e = st->buf.find("\r\n\r\n");
        if (e == std::string::npos) return;
        st->ok = st->buf.compare(0, 12, "HTTP/1.1 200") == 0;
        st->buf.erase(0, e + 4);
        st->head = true;
    }
    if (!st->ok) return;   // an error body: read once complete
    for (size_t e; (e = st->buf.find("\n\n")) != std::string::npos;) {
        std::string ev = st->buf.substr(0, e);
        st->buf.erase(0, e + 2);
        if (ev.compare(0, 6, "data: ") || ev == "data: [DONE]") continue;
        char err[96];
        Json *j = json_parse(ev.data() + 6, (int)ev.size() - 6, err, sizeof(err));
        Json *choices = json_get(j, "choices"), *delta = choices && choices->child ? json_get(choices->child, "delta") : nullptr;
        std::string content = json_str(json_get(delta, "content"), ""), reasoning = json_str(json_get(delta, "reasoning_content"), "");
        std::string error = json_get(j, "error") ? json_str(json_get(json_get(j, "error"), "message"), "error") : "";
        json_free(j);
        if (!content.empty() || !reasoning.empty() || !error.empty()) st->on_delta(content, reasoning, error);
    }
}

// The message of a non-200 response body ({"error":{"message":...}}).
static std::string error_message(const std::string &body) {
    char err[96];
    Json *j = json_parse(body.data(), (int)body.size(), err, sizeof(err));
    std::string m = json_str(json_get(json_get(j, "error"), "message"), "the server returned an error");
    json_free(j);
    return m;
}

/* ------------------------------------------------------------------ screen */

struct App : ApplicationT<App> {
    std::wstring m_local, m_ip, m_info_text;
    std::string m_log_raw, m_cpu, m_kernel;
    bool m_ready = false, m_log_follow = true;
    int m_shown = -1;
    std::chrono::steady_clock::time_point m_since;
    Windows::UI::Core::CoreDispatcher m_dispatcher{ nullptr };
    DispatcherTimer m_timer{ nullptr };
    Border m_pill{ nullptr };
    Shapes::Ellipse m_dot{ nullptr };
    TextBlock m_status{ nullptr }, m_info{ nullptr }, m_log{ nullptr };
    ScrollViewer m_log_scroll{ nullptr }, m_chat_scroll{ nullptr };

    // chat: past turns (role, text), the reply being generated
    std::vector<std::pair<std::string, std::string>> m_history;
    StackPanel m_chat_list{ nullptr };
    TextBox m_input{ nullptr };
    Button m_send{ nullptr }, m_new{ nullptr };
    TextBlock m_hint{ nullptr };
    Documents::Run m_reason_run{ nullptr }, m_answer_run{ nullptr };
    std::wstring m_reason, m_answer;
    bool m_busy = false;
    volatile int m_stop = 0;

    App() {
        RequestedTheme(ApplicationTheme::Dark);
        // the gamepad moves the focus between controls instead of driving a mouse pointer
        RequiresPointerMode(ApplicationRequiresPointerMode::WhenRequested);
    }

    void OnLaunched(Windows::ApplicationModel::Activation::LaunchActivatedEventArgs const &) {
        Window window = Window::Current();
        if (!window.Content()) {
            m_dispatcher = window.Dispatcher();
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
        root.RowSpacing(14);
        rows(root, { fit(), star() });

        // header: mark, name, state, stop button
        Grid head;
        head.ColumnSpacing(16);
        cols(head, { fit(), star(), fit(), fit() });
        if (auto mark = logo(52)) place(head, mark, 0, 0);
        StackPanel title;
        title.VerticalAlignment(VerticalAlignment::Center);
        TextBlock name = text(L"", 32, TEXT, true);
        Documents::Run word, dot;
        word.Text(L"pithos");
        dot.Text(L".");
        dot.Foreground(brush(GREEN));
        name.Inlines().Append(word);
        name.Inlines().Append(dot);
        title.Children().Append(name);
        title.Children().Append(text(L"OpenAI-compatible LLM server on Xbox", 13, MUTED));
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

        // body: chat (left), how to connect and the server log (right)
        Grid body;
        body.ColumnSpacing(14);
        cols(body, { star(3), star(2) });
        place(body, panel(build_chat()), 0, 0);

        Grid side;
        side.RowSpacing(14);
        rows(side, { fit(), star() });
        StackPanel info;
        info.Spacing(8);
        info.Children().Append(text(L"Connect", 17, TEXT, true));
        m_info = text(L"", 13, TEXT);
        m_info.LineHeight(21);
        info.Children().Append(m_info);
        place(side, panel(info), 0);

        Grid log;
        log.RowSpacing(8);
        rows(log, { fit(), star() });
        place(log, text(L"Server log", 17, TEXT, true), 0);
        m_log = text(L"", 11, LOGTEXT);
        m_log.FontFamily(FontFamily(L"Consolas, Courier New"));
        m_log_scroll = ScrollViewer();
        m_log_scroll.Content(m_log);
        m_log_scroll.IsTabStop(true);   // focusable, so the gamepad can scroll it
        m_log_scroll.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
        place(log, m_log_scroll, 1);
        place(side, panel(log), 1);
        place(body, side, 0, 1);
        place(root, body, 1);
        return root;
    }

    Grid build_chat() {
        Grid chat;
        chat.RowSpacing(10);
        rows(chat, { fit(), star(), fit() });

        Grid top;
        cols(top, { star(), fit() });
        StackPanel names;
        names.Children().Append(text(L"Chat", 17, TEXT, true));
        names.Children().Append(text(L"Talks to the model on this console. No commands or tools.", 12, MUTED));
        place(top, names, 0, 0);
        m_new = Button();
        m_new.Content(box_value(L"New chat"));
        m_new.VerticalAlignment(VerticalAlignment::Center);
        m_new.Click([this](auto &&, auto &&) { new_chat(); });
        place(top, m_new, 0, 1);
        place(chat, top, 0);

        m_chat_list = StackPanel();
        m_chat_list.Spacing(10);
        m_chat_list.Padding(Thickness{ 0, 0, 12, 0 });
        m_hint = text(L"Select the box below and press A to type. The answer appears here.", 15, MUTED);
        m_hint.Margin(Thickness{ 0, 12, 0, 0 });
        m_chat_list.Children().Append(m_hint);
        m_chat_scroll = ScrollViewer();
        m_chat_scroll.Content(m_chat_list);
        m_chat_scroll.IsTabStop(true);
        m_chat_scroll.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
        place(chat, m_chat_scroll, 1);

        Grid bottom;
        bottom.ColumnSpacing(10);
        cols(bottom, { star(), fit() });
        m_input = TextBox();
        m_input.PlaceholderText(L"Write a message");
        m_input.FontSize(16);
        m_input.KeyDown([this](auto &&, Input::KeyRoutedEventArgs const &e) {
            if (e.Key() == Windows::System::VirtualKey::Enter) {
                e.Handled(true);
                send();
            }
        });
        place(bottom, m_input, 0, 0);
        m_send = Button();
        m_send.Content(box_value(L"Send"));
        m_send.MinWidth(96);
        m_send.VerticalAlignment(VerticalAlignment::Stretch);
        m_send.Click([this](auto &&, auto &&) {
            if (m_busy) m_stop = 1;
            else send();
        });
        place(bottom, m_send, 0, 1);
        place(chat, bottom, 2);
        // start next to the box: focusing the box itself would open the on-screen keyboard at launch
        m_send.Loaded([this](auto &&, auto &&) { m_send.Focus(FocusState::Programmatic); });
        return chat;
    }

    // A message bubble; the caller fills the returned text block.
    TextBlock bubble(bool user) {
        if (m_hint) {
            uint32_t i;
            if (m_chat_list.Children().IndexOf(m_hint, i)) m_chat_list.Children().RemoveAt(i);
        }
        TextBlock t = text(L"", 16, TEXT);
        t.LineHeight(24);
        Border b;
        b.Child(t);
        b.CornerRadius(CornerRadius{ 10, 10, 10, 10 });
        b.Padding(Thickness{ 14, 9, 14, 10 });
        b.MaxWidth(520);
        b.Background(brush(user ? 0x0f4d10 : 0x232923));
        b.HorizontalAlignment(user ? HorizontalAlignment::Right : HorizontalAlignment::Left);
        m_chat_list.Children().Append(b);
        return t;
    }

    void scroll_chat() {
        m_chat_scroll.UpdateLayout();
        m_chat_scroll.ChangeView(nullptr, IReference<double>(m_chat_scroll.ScrollableHeight()), nullptr);
    }

    void send() {
        if (m_busy || m_shown != RUNNING) return;
        std::wstring w(m_input.Text());
        w.erase(0, w.find_first_not_of(L" \t\r\n"));
        w.erase(w.find_last_not_of(L" \t\r\n") + 1);
        if (w.empty()) return;
        m_input.Text(L"");
        bubble(true).Text(w);
        m_history.emplace_back("user", utf8(w));

        TextBlock reply = bubble(false);
        m_reason_run = Documents::Run();
        m_reason_run.Foreground(brush(MUTED));
        m_reason_run.FontStyle(Windows::UI::Text::FontStyle::Italic);
        m_reason_run.Text(L"...");
        m_answer_run = Documents::Run();
        reply.Inlines().Append(m_reason_run);
        reply.Inlines().Append(m_answer_run);
        m_reason.clear();
        m_answer.clear();
        m_busy = true;
        m_stop = 0;
        m_send.Content(box_value(L"Stop"));
        m_new.IsEnabled(false);
        scroll_chat();

        Buf b = {};
        buf_add(&b, "{\"stream\":true,\"messages\":[", -1);
        for (size_t i = 0; i < m_history.size(); i++) {
            buf_printf(&b, "%s{\"role\":\"%s\",\"content\":", i ? "," : "", m_history[i].first.c_str());
            buf_json_str(&b, m_history[i].second.data(), (int)m_history[i].second.size());
            buf_add(&b, "}", 1);
        }
        buf_add(&b, "]}", 2);
        std::string body(b.s, b.n);
        buf_free(&b);
        std::thread([this, body] { generate(body); }).detach();
    }

    // Worker thread: runs the request on the engine (it waits its turn behind network requests).
    void generate(std::string body) {
        ChatStream st;
        st.on_delta = [this](std::string c, std::string r, std::string e) {
            m_dispatcher.RunAsync(Windows::UI::Core::CoreDispatcherPriority::Normal,
                                  [this, c = wide(c), r = wide(r), e = wide(e)] { on_delta(c, r, e); });
        };
        server_local_chat(body.data(), (int)body.size(), chat_sink, &st, &m_stop);
        std::wstring err = st.ok ? L"" : wide(error_message(st.buf));
        m_dispatcher.RunAsync(Windows::UI::Core::CoreDispatcherPriority::Normal, [this, err] { finish(err); });
    }

    void on_delta(std::wstring const &c, std::wstring const &r, std::wstring const &e) {
        m_reason += r;
        m_answer += c;
        if (!e.empty()) m_answer += L"\n[" + e + L"]";
        size_t start = m_answer.find_first_not_of(L"\r\n");
        m_reason_run.Text(m_reason.empty() ? L"" : m_reason + (m_answer.empty() ? L"" : L"\n\n"));
        m_answer_run.Text(start == std::wstring::npos ? L"" : m_answer.substr(start));
        if (m_reason.empty() && m_answer.empty()) m_reason_run.Text(L"...");
        bool at_end = m_chat_scroll.VerticalOffset() >= m_chat_scroll.ScrollableHeight() - 40;
        if (at_end) scroll_chat();
    }

    void finish(std::wstring const &err) {
        std::wstring answer = m_answer;
        answer.erase(0, answer.find_first_not_of(L" \r\n"));
        if (!err.empty()) m_answer_run.Text(L"[" + err + L"]");
        else if (m_stop) m_answer_run.Text(answer + L" [stopped]");
        if (m_reason.empty() && answer.empty() && err.empty()) m_reason_run.Text(L"");
        // a turn without an answer is dropped, so the history keeps alternating
        if (answer.empty()) m_history.pop_back();
        else m_history.emplace_back("assistant", utf8(answer));
        m_busy = false;
        m_send.Content(box_value(L"Send"));
        m_new.IsEnabled(true);
        scroll_chat();
    }

    void new_chat() {
        if (m_busy) return;
        m_history.clear();
        m_chat_list.Children().Clear();
        m_chat_list.Children().Append(m_hint);
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
            m_log_follow = m_log_scroll.VerticalOffset() >= m_log_scroll.ScrollableHeight() - 30;
            m_log.Text(clean_log(raw, 400));
        }
        // keep following the end while the user has not scrolled up (the layout of new text
        // can finish after this tick, so it is repeated on every one)
        if (m_log_follow)
            m_log_scroll.ChangeView(nullptr, IReference<double>(m_log_scroll.ScrollableHeight()), nullptr, true);
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

    // Connect panel: one "label  value" line per entry, rebuilt only when the text changes.
    void show_info(int state) {
        std::vector<std::pair<std::wstring, std::wstring>> lines;
        if (state == WAITING) {
            lines = { { L"", L"Upload a .gguf model into the app's LocalState folder from the Device Portal:" },
                      { L"", L"python xbox/deploy.py --xbox " + m_ip + L" upload MODEL.gguf" },
                      { L"", L"The server starts as soon as the upload is complete." } };
        } else {
            wchar_t gb[32];
            swprintf(gb, 32, L" (%.2f GB)", g_model_gb);
            lines = { { L"Web chat", L"http://" + m_ip + L":8080" },
                      { L"OpenAI API", L"http://" + m_ip + L":8080/v1" },
                      { L"Model", g_model + gb } };
            if (!m_cpu.empty()) lines.push_back({ L"Hardware", wide(m_cpu.substr(0, m_cpu.find(" | "))) });
            if (!m_kernel.empty()) lines.push_back({ L"Kernels", wide(m_kernel) });
            if (state == RUNNING) {
                long long m = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - m_since).count();
                wchar_t up[48];
                swprintf(up, 48, L"%lld h %02lld min", m / 60, m % 60);
                lines.push_back({ L"Running for", up });
            }
        }
        std::wstring key;
        for (auto &l : lines) key += l.first + L"\t" + l.second + L"\n";
        if (key == m_info_text) return;
        m_info_text = key;
        m_info.Inlines().Clear();
        for (size_t i = 0; i < lines.size(); i++) {
            if (i) m_info.Inlines().Append(Documents::LineBreak());
            if (!lines[i].first.empty()) {
                Documents::Run label;
                label.Text(lines[i].first + L"   ");
                label.Foreground(brush(MUTED));
                m_info.Inlines().Append(label);
            }
            Documents::Run value;
            value.Text(lines[i].second);
            m_info.Inlines().Append(value);
        }
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
