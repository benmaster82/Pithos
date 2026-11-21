// talk.cpp - the full-screen voice mode (talk.h).
//
// A 1600 x 900 design canvas in a Viewbox: a deep black background; a layered neural network on
// both sides of the centre, whose pulses run along its edges and light up the nodes they reach
// (rare at rest, dense while the model thinks, with the voice while it listens or speaks); at
// the centre the Pithos mark with its blinking eye (as web/talk.html), an aura and a ring that
// follow the voice level, a ring of ticks and two arcs turning the opposite ways; around it the
// time, the model's speed, the state, the subtitles, the voice language and the controls.
#include "talk.h"
#include <math.h>
#include <chrono>
#include <ctime>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.UI.h>
#include <winrt/Windows.UI.Text.h>
#include <winrt/Windows.UI.Xaml.Markup.h>
#include <winrt/Windows.UI.Xaml.Media.h>

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Media;

namespace {

const double W = 1600, H = 900, CX = 800, CY = 385;   // the design canvas and the centre of the HUD
const uint32_t DEEP = 0x020305, ICE = 0xa9cfe0, ICE_DIM = 0x5f8496, ICE_HI = 0xdff3fb, AMBER = 0xe8ad4f,
               TEXT = 0xe6f2f8, MUTED = 0x7d8e9c, MARK = 0xeef6fa;

Windows::UI::Color col(uint32_t rgb, uint8_t a = 255) {
    return Windows::UI::ColorHelper::FromArgb(a, (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255);
}
SolidColorBrush solid(uint32_t rgb, uint8_t a = 255) { return SolidColorBrush(col(rgb, a)); }

uint32_t state_color(TalkState s) {
    switch (s) {
    case TALK_LISTENING: return 0x9fe6ff;
    case TALK_TRANSCRIBING: return ICE;
    case TALK_THINKING: case TALK_WRITING: return AMBER;
    case TALK_SPEAKING: return ICE_HI;
    default: return ICE_DIM;
    }
}

const wchar_t *state_name(TalkState s) {
    switch (s) {
    case TALK_LISTENING: return L"\x25cf  LISTENING";
    case TALK_TRANSCRIBING: return L"\x25cf  TRANSCRIBING";
    case TALK_THINKING: return L"\x25cf  THINKING";
    case TALK_WRITING: return L"\x25cf  ANSWERING";
    case TALK_SPEAKING: return L"\x25b6  SPEAKING";
    default: return L"\x25cb  READY";
    }
}

TextBlock label(double size, uint32_t color, bool bold = false) {
    TextBlock t;
    t.FontSize(size);
    t.Foreground(solid(color));
    if (bold) t.FontWeight(Windows::UI::Text::FontWeights::SemiBold());
    return t;
}

void at(UIElement const &e, double x, double y) {
    Canvas::SetLeft(e, x);
    Canvas::SetTop(e, y);
}

Shapes::Path path(std::wstring const &data, uint32_t stroke, uint8_t alpha, double thickness) {
    std::wstring xaml = L"<Path xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation' Data='" + data + L"'/>";
    Shapes::Path p = Markup::XamlReader::Load(xaml).as<Shapes::Path>();
    p.Stroke(solid(stroke, alpha));
    p.StrokeThickness(thickness);
    p.StrokeStartLineCap(PenLineCap::Round);
    p.StrokeEndLineCap(PenLineCap::Round);
    return p;
}

std::wstring num(double v) {
    wchar_t b[32];
    swprintf(b, 32, L"%.1f", v);
    return b;
}

// an arc of radius r around (c, c) from angle a0 to a1 (degrees, clockwise from the top)
std::wstring arc(double c, double r, double a0, double a1) {
    const double k = 3.14159265358979 / 180;
    const double x0 = c + r * sin(a0 * k), y0 = c - r * cos(a0 * k), x1 = c + r * sin(a1 * k), y1 = c - r * cos(a1 * k);
    return L"M " + num(x0) + L"," + num(y0) + L" A " + num(r) + L"," + num(r) + L" 0 " + (a1 - a0 > 180 ? L"1" : L"0") +
           L" 1 " + num(x1) + L"," + num(y1);
}

void rotate(FrameworkElement const &e, double angle) {
    RotateTransform r = e.RenderTransform().try_as<RotateTransform>();
    if (!r) {
        r = RotateTransform();
        e.RenderTransform(r);
        e.RenderTransformOrigin(Point{ 0.5f, 0.5f });
    }
    r.Angle(angle);
}

void scale(FrameworkElement const &e, double s) {
    ScaleTransform t = e.RenderTransform().try_as<ScaleTransform>();
    if (!t) {
        t = ScaleTransform();
        e.RenderTransform(t);
        e.RenderTransformOrigin(Point{ 0.5f, 0.5f });
    }
    t.ScaleX(s);
    t.ScaleY(s);
}

}   // namespace

static double rnd01(uint64_t *r) {   // xorshift64*
    uint64_t x = *r;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *r = x;
    return (double)((x * 0x2545F4914F6CDD1Dull) >> 11) / 9007199254740992.0;
}

void TalkView::build_network(Canvas const &c) {
    // three columns on each side, then anchors on a circle around the HUD where the inner
    // columns end; edges run from each node to two or three of the next column inwards
    const double xs[6] = { 120, 300, 465, 1135, 1300, 1480 };
    const int counts[6] = { 6, 8, 7, 7, 8, 6 };
    std::vector<std::vector<int>> cols(6);
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < counts[k]; i++) {
            const double y = 150 + (500.0 * (i + 0.5)) / counts[k] + (rnd01(&m_rng) - 0.5) * 40;
            cols[k].push_back((int)m_nodes.size());
            m_nodes.push_back({ xs[k] + (rnd01(&m_rng) - 0.5) * 30, y, 0 });
        }
    std::vector<int> left_anchor, right_anchor;
    for (int i = 0; i < 5; i++) {   // on the circle of radius 250, at the left and at the right
        const double a = (-50 + 25 * i) * 3.14159265358979 / 180;
        left_anchor.push_back((int)m_nodes.size());
        m_nodes.push_back({ CX - 250 * cos(a), CY + 250 * sin(a), 0 });
        right_anchor.push_back((int)m_nodes.size());
        m_nodes.push_back({ CX + 250 * cos(a), CY + 250 * sin(a), 0 });
    }
    auto link = [&](const std::vector<int> &from, const std::vector<int> &to) {
        for (int a : from) {
            const int n = 2 + (int)(rnd01(&m_rng) * 2);
            for (int j = 0; j < n; j++) {
                const int b = to[(size_t)(rnd01(&m_rng) * to.size()) % to.size()];
                bool dup = false;
                for (auto &e : m_edges) dup = dup || (e.a == a && e.b == b);
                if (!dup) m_edges.push_back({ a, b });
            }
        }
    };
    link(cols[0], cols[1]); link(cols[1], cols[2]); link(cols[2], left_anchor);
    link(cols[5], cols[4]); link(cols[4], cols[3]); link(cols[3], right_anchor);

    SolidColorBrush edge = solid(ICE, 0x1c), node = solid(ICE);
    for (auto &e : m_edges) {
        e.l = Shapes::Line();
        e.l.X1(m_nodes[e.a].x); e.l.Y1(m_nodes[e.a].y);
        e.l.X2(m_nodes[e.b].x); e.l.Y2(m_nodes[e.b].y);
        e.l.Stroke(edge);
        e.l.StrokeThickness(1.2);
        c.Children().Append(e.l);
    }
    for (size_t i = 0; i < m_nodes.size(); i++) {
        auto &n = m_nodes[i];
        const bool anchor = i >= m_nodes.size() - 10;
        const double d = anchor ? 5 : 8 + rnd01(&m_rng) * 4;
        n.e = Shapes::Ellipse();
        n.e.Width(d);
        n.e.Height(d);
        n.e.Fill(node);
        n.e.Opacity(0.2);
        at(n.e, n.x - d / 2, n.y - d / 2);
        c.Children().Append(n.e);
    }
    m_pulse_brush = solid(ICE_DIM);
    for (int i = 0; i < 48; i++) {
        Pulse p;
        p.e = Shapes::Ellipse();
        p.e.Width(7);
        p.e.Height(7);
        p.e.Fill(m_pulse_brush);
        p.e.Visibility(Visibility::Collapsed);
        c.Children().Append(p.e);
        m_pulses.push_back(p);
    }
}

void TalkView::step_network(double dt, double activity) {
    // new pulses, activity a second, on random edges (half of them outwards)
    m_spawn += activity * dt;
    while (m_spawn >= 1) {
        m_spawn -= 1;
        for (auto &p : m_pulses)
            if (p.edge < 0) {
                p.edge = (int)(rnd01(&m_rng) * m_edges.size()) % (int)m_edges.size();
                p.speed = (0.8 + rnd01(&m_rng) * 0.9) * (rnd01(&m_rng) < 0.5 ? 1 : -1);
                p.t = p.speed > 0 ? 0 : 1;
                p.e.Visibility(Visibility::Visible);
                break;
            }
    }
    for (auto &p : m_pulses) {
        if (p.edge < 0) continue;
        p.t += p.speed * dt;
        const Edge &e = m_edges[p.edge];
        if (p.t >= 1 || p.t <= 0) {   // arrived: the node it reached lights up
            m_nodes[p.t >= 1 ? e.b : e.a].glow = 1;
            p.edge = -1;
            p.e.Visibility(Visibility::Collapsed);
            continue;
        }
        const Node &a = m_nodes[e.a], &c = m_nodes[e.b];
        p.e.Opacity(0.35 + 0.65 * sin(3.14159265358979 * p.t));
        at(p.e, a.x + (c.x - a.x) * p.t - 3.5, a.y + (c.y - a.y) * p.t - 3.5);
    }
    for (auto &n : m_nodes) {
        n.glow *= exp(-dt * 2.5);
        n.e.Opacity(0.16 + 0.84 * n.glow);
    }
}

UIElement TalkView::build(Shapes::Path mark) {
    Grid root;
    root.Background(solid(DEEP));
    Viewbox vb;
    vb.Stretch(Stretch::Uniform);
    Canvas c;
    c.Width(W);
    c.Height(H);
    vb.Child(c);
    root.Children().Append(vb);

    Canvas net;
    net.Width(W);
    net.Height(H);
    c.Children().Append(net);
    build_network(net);

    // the HUD: a 520 x 520 square around the centre
    const double S = 520, h = S / 2;
    Grid hud;
    hud.Width(S);
    hud.Height(S);
    at(hud, CX - h, CY - h);
    c.Children().Append(hud);

    // the aura: translucent discs one inside the other, a glow brighter towards the centre (system
    // XAML has no radial gradient)
    Grid aura;
    aura.Width(440);
    aura.Height(440);
    m_aura_brush = solid(ICE_DIM, 0x12);
    for (int i = 0; i < 7; i++) {
        Shapes::Ellipse e;
        e.Width(440 - i * 38);
        e.Height(440 - i * 38);
        e.Fill(m_aura_brush);
        aura.Children().Append(e);
    }
    m_aura = aura;
    hud.Children().Append(aura);

    m_ring = Shapes::Ellipse();
    m_ring.Width(330);
    m_ring.Height(330);
    m_ring.Stroke(solid(ICE, 0x90));
    m_ring.StrokeThickness(2);
    hud.Children().Append(m_ring);

    std::wstring ticks;   // 90 ticks, a longer one every 15
    for (int i = 0; i < 90; i++) {
        const double a = i * 4 * 3.14159265358979 / 180, r0 = i % 15 ? 238 : 230, r1 = 246;
        ticks += L"M " + num(h + r0 * sin(a)) + L"," + num(h - r0 * cos(a)) + L" L " + num(h + r1 * sin(a)) + L"," +
                 num(h - r1 * cos(a)) + L" ";
    }
    m_ticks = path(ticks, ICE, 0x60, 1.5);
    m_ticks.Width(S);
    m_ticks.Height(S);
    hud.Children().Append(m_ticks);
    m_arc1 = path(arc(h, 196, 20, 130) + L" " + arc(h, 196, 200, 310), ICE, 0xb0, 3);
    m_arc1.Width(S);
    m_arc1.Height(S);
    hud.Children().Append(m_arc1);
    m_arc2 = path(arc(h, 214, 100, 160) + L" " + arc(h, 214, 280, 340), ICE, 0x70, 2);
    m_arc2.Width(S);
    m_arc2.Height(S);
    hud.Children().Append(m_arc2);

    // the mark and its eyelid (web/talk.html): a 100 x 100 drawing at 2.1 x
    const double M = 210, k = M / 100;
    Grid box;
    box.Width(M);
    box.Height(M);
    mark.RenderTransform(nullptr);
    mark.Width(M);
    mark.Height(M);
    mark.Fill(solid(MARK));
    box.Children().Append(mark);
    Canvas lidc;
    lidc.Width(M);
    lidc.Height(M);
    m_lid = Shapes::Ellipse();
    m_lid.Width(2 * 11.2 * k);
    m_lid.Height(2 * 12 * k);
    m_lid.Fill(solid(MARK));
    at(m_lid, (68.49 - 11.2) * k, (57.93 - 12) * k);
    TransformGroup lt;
    ScaleTransform blink;
    blink.ScaleY(0);
    blink.CenterX(11.2 * k);
    blink.CenterY(12 * k);
    RotateTransform tilt;
    tilt.Angle(10.8);
    tilt.CenterX(11.2 * k);
    tilt.CenterY(12 * k);
    lt.Children().Append(blink);
    lt.Children().Append(tilt);
    m_lid.RenderTransform(lt);
    lidc.Children().Append(m_lid);
    box.Children().Append(lidc);
    TransformGroup mt;   // the mark leans as in the header; its scale follows the voice
    ScaleTransform ms;
    RotateTransform mr;
    mr.Angle(-9);
    mt.Children().Append(ms);
    mt.Children().Append(mr);
    box.RenderTransform(mt);
    box.RenderTransformOrigin(Point{ 0.5f, 0.5f });
    m_markbox = box;
    hud.Children().Append(box);

    // top: time and date at the left, the name in the middle, the speed at the right
    m_clock = label(44, TEXT);
    m_clock.FontWeight(Windows::UI::Text::FontWeights::Light());
    at(m_clock, 60, 34);
    c.Children().Append(m_clock);
    m_date = label(16, MUTED);
    at(m_date, 62, 92);
    c.Children().Append(m_date);
    TextBlock name = label(18, ICE, true);
    name.Text(L"P I T H O S . X");
    name.Width(400);
    name.TextAlignment(TextAlignment::Center);
    at(name, CX - 200, 52);
    c.Children().Append(name);
    m_speed = label(16, MUTED);
    m_speed.Width(360);
    m_speed.TextAlignment(TextAlignment::Right);
    at(m_speed, W - 60 - 360, 56);
    c.Children().Append(m_speed);

    // the subtitles under the HUD, always shown
    m_user = label(22, ICE);
    m_user.Width(1300);
    m_user.TextAlignment(TextAlignment::Center);
    m_user.TextWrapping(TextWrapping::Wrap);
    m_user.MaxLines(1);
    m_user.TextTrimming(TextTrimming::CharacterEllipsis);
    at(m_user, CX - 650, 648);
    c.Children().Append(m_user);
    m_answer = label(30, TEXT);
    m_answer.Width(1300);
    m_answer.TextAlignment(TextAlignment::Center);
    m_answer.TextWrapping(TextWrapping::Wrap);
    m_answer.MaxLines(3);
    m_answer.LineHeight(38);
    at(m_answer, CX - 650, 684);
    c.Children().Append(m_answer);

    // bottom: the state, the voice and reasoning, the controls
    m_status = label(18, ICE_DIM, true);
    m_status.CharacterSpacing(120);
    at(m_status, 60, 846);
    c.Children().Append(m_status);
    m_info = label(15, MUTED);
    at(m_info, 370, 849);
    c.Children().Append(m_info);
    m_hint = label(15, MUTED);
    m_hint.Width(900);
    m_hint.TextAlignment(TextAlignment::Right);
    at(m_hint, W - 60 - 900, 849);
    c.Children().Append(m_hint);
    return root;
}

void TalkView::update(const TalkInfo &in) {
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    double dt = m_last < 0 ? 0.033 : now - m_last;
    if (dt > 0.2) dt = 0.2;
    m_last = now;
    m_time += dt;

    const uint32_t accent = state_color(in.state);
    const std::wstring status = in.state == TALK_READY && in.hands_free ? L"\x25c9  HANDS-FREE" : state_name(in.state);
    if (status != std::wstring(m_status.Text())) m_status.Text(status);
    if (in.state != m_state) {
        m_state = in.state;
        m_status.Foreground(solid(accent));
        m_pulse_brush.Color(col(accent));
        m_ring.Stroke(solid(accent, 0x90));
        m_arc1.Stroke(solid(accent, 0xb0));
        m_arc2.Stroke(solid(accent, 0x70));
        m_aura_brush.Color(col(accent, 0x12));
    }

    // the voice level, rising fast and falling slowly (web/talk.html)
    const double target = in.state == TALK_LISTENING || in.state == TALK_SPEAKING ? in.level : 0;
    m_level += (target - m_level) * (target > m_level ? 0.5 : 0.15);
    const double breathe = in.state == TALK_READY ? 0.03 * sin(m_time * 1.25) : 0;
    scale(m_aura, 0.82 + m_level * 0.55 + breathe);
    scale(m_ring, 1 + m_level * 0.25 + breathe * 0.5);
    m_ring.Opacity(0.35 + m_level * 0.65);
    auto mt = m_markbox.RenderTransform().as<TransformGroup>();
    auto ms = mt.Children().GetAt(0).as<ScaleTransform>();
    const double ponder = in.state == TALK_THINKING ? 0.02 * sin(m_time * 4) : 0;
    ms.ScaleX(1 + m_level * 0.1 + breathe * 0.5 + ponder);
    ms.ScaleY(1 + m_level * 0.1 + breathe * 0.5 + ponder);

    // the arcs turn, faster while it thinks
    const double spin = in.state == TALK_THINKING || in.state == TALK_WRITING ? 110
                      : in.state == TALK_TRANSCRIBING ? 70
                      : in.state == TALK_SPEAKING ? 25 + 80 * m_level
                      : in.state == TALK_LISTENING ? 35 + 60 * m_level : 7;
    static double a1, a2;
    a1 += spin * dt;
    a2 -= spin * 0.6 * dt;
    rotate(m_arc1, a1);
    rotate(m_arc2, a2);
    rotate(m_ticks, a2 * 0.15);

    // the eye blinks every few seconds, for 140 ms
    if (m_time >= m_blink_at) {
        m_blink = 0.14;
        m_blink_at = m_time + 2.5 + rnd01(&m_rng) * 3.5;
    }
    m_blink -= dt;
    m_lid.RenderTransform().as<TransformGroup>().Children().GetAt(0).as<ScaleTransform>().ScaleY(m_blink > 0 ? 1 : 0);

    const double activity = in.state == TALK_THINKING ? 16 : in.state == TALK_WRITING ? 11 : in.state == TALK_TRANSCRIBING ? 8
                          : in.state == TALK_SPEAKING ? 3 + 14 * m_level : in.state == TALK_LISTENING ? 2 + 16 * m_level : 0.7;
    step_network(dt, activity);

    // the clock, once a second
    static time_t shown;
    const time_t t = time(nullptr);
    if (t != shown) {
        shown = t;
        tm lt;
        localtime_s(&lt, &t);
        wchar_t b[64];
        wcsftime(b, 64, L"%H:%M", &lt);
        m_clock.Text(b);
        wcsftime(b, 64, L"%A %d %B %Y", &lt);
        m_date.Text(b);
    }
    if (std::wstring(m_speed.Text()) != in.speed) m_speed.Text(in.speed);

    // the subtitles: the question, and the end of the answer as it grows
    if (in.user != m_shown_user) {
        m_shown_user = in.user;
        m_user.Text(in.user.empty() ? L"" : L"\x201c" + in.user + L"\x201d");
    }
    if (in.answer != m_shown_answer) {
        m_shown_answer = in.answer;
        std::wstring a = in.answer;
        const size_t cap = 240;
        if (a.size() > cap) {
            size_t s = a.find(L' ', a.size() - cap);
            a = L"\x2026" + a.substr(s == std::wstring::npos ? a.size() - cap : s);
        }
        m_answer.Text(a);
    }
    std::wstring info = L"Voice  " + in.voice + L"     Reasoning  " + in.reasoning;
    if (!in.note.empty()) info = in.note;
    if (std::wstring(m_info.Text()) != info) m_info.Text(info);
    const wchar_t *hint = !in.can_listen ? L"Y  repeat     LB RB  voice     \x2191\x2193  reasoning     B  back"
                        : in.hands_free ? L"just speak     X  hands-free off     Y  repeat     LB RB  voice     \x2191\x2193  reasoning     B  back"
                                        : L"RT  hold to talk     X  hands-free     Y  repeat     LB RB  voice     \x2191\x2193  reasoning     B  back";
    if (std::wstring(m_hint.Text()) != hint) m_hint.Text(hint);
}
