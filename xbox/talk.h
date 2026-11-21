// talk.h - the full-screen voice mode of the Xbox app (talk.cpp): the Pithos mark at the centre of a
// HUD, an animated neural network behind it, the state, the subtitles and the controls.
#pragma once
#include <string>
#include <vector>
#include <winrt/Windows.UI.Xaml.h>
#include <winrt/Windows.UI.Xaml.Controls.h>
#include <winrt/Windows.UI.Xaml.Media.h>
#include <winrt/Windows.UI.Xaml.Shapes.h>

enum TalkState { TALK_READY, TALK_LISTENING, TALK_TRANSCRIBING, TALK_THINKING, TALK_WRITING, TALK_SPEAKING };

// what the screen shows, set by the app on every frame
struct TalkInfo {
    TalkState state = TALK_READY;
    float level = 0;              // 0..1: the voice heard (listening) or spoken (speaking)
    std::wstring user, answer;    // the last question and its answer (the subtitles)
    std::wstring speed;           // e.g. "29.5 tok/s"
    std::wstring voice, reasoning;
    std::wstring note;            // a message for a few seconds, or ""
    bool can_listen = true;       // a microphone is there
};

class TalkView {
public:
    // mark: the Pithos mark as a XAML path (its own transform is replaced)
    winrt::Windows::UI::Xaml::UIElement build(winrt::Windows::UI::Xaml::Shapes::Path mark);
    void update(const TalkInfo &info);   // about 30 times a second while shown

private:
    struct Node { double x, y, glow; winrt::Windows::UI::Xaml::Shapes::Ellipse e{ nullptr }; };
    struct Edge { int a, b; winrt::Windows::UI::Xaml::Shapes::Line l{ nullptr }; };
    struct Pulse { int edge = -1; double t = 0, speed = 1; winrt::Windows::UI::Xaml::Shapes::Ellipse e{ nullptr }; };

    void build_network(winrt::Windows::UI::Xaml::Controls::Canvas const &c);
    void step_network(double dt, double activity);

    std::vector<Node> m_nodes;
    std::vector<Edge> m_edges;
    std::vector<Pulse> m_pulses;
    double m_spawn = 0, m_time = 0, m_level = 0, m_blink_at = 3, m_blink = 0;
    uint64_t m_rng = 0x9E3779B97F4A7C15ull;
    TalkState m_state = TALK_READY;
    double m_last = -1;

    winrt::Windows::UI::Xaml::Media::SolidColorBrush m_pulse_brush{ nullptr }, m_aura_brush{ nullptr };   // the state's colour
    winrt::Windows::UI::Xaml::FrameworkElement m_aura{ nullptr };
    winrt::Windows::UI::Xaml::Shapes::Ellipse m_ring{ nullptr }, m_lid{ nullptr };
    winrt::Windows::UI::Xaml::Shapes::Path m_arc1{ nullptr }, m_arc2{ nullptr }, m_ticks{ nullptr };
    winrt::Windows::UI::Xaml::FrameworkElement m_markbox{ nullptr };
    winrt::Windows::UI::Xaml::Controls::TextBlock m_clock{ nullptr }, m_date{ nullptr }, m_speed{ nullptr },
        m_status{ nullptr }, m_hint{ nullptr }, m_user{ nullptr }, m_answer{ nullptr }, m_info{ nullptr };
    std::wstring m_shown_user, m_shown_answer;
};
