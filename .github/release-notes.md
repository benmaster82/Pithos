Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B and
LFM3-1.7B locally, with a terminal chat, the Pandora web chat, an OpenAI-compatible server and a
terminal agent. This release adds speech: with LFM3-Audio-2B, Pithos hears, speaks and holds voice
conversations.

## Downloads

| File | For |
|---|---|
| `pithos-0.7.0-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.7.0-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.7.0-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB, LFM3-Audio-2B 1.3 GB with its audio projector), put it in a
`models` folder next to the program, then run for example `pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.6.2

**LFM3-Audio-2B (experimental).** Liquid AI's speech model: LFM3 with a speech encoder and a speech
decoder, in 18 languages. It needs two files from
[LiquidAI/LFM3-Audio-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF): the model and its
audio projector (`mmproj-...`), which Pithos finds next to the model. On Linux,
`bash build.sh --download-model=lfm3-audio-2b` downloads and verifies both.

- **Speech input.** `--audio speech.wav` on the command line; in the server, `input_audio` parts in
  chat messages, `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` (speech to English
  text). WAV input, any sample rate. Transcriptions match llama.cpp's; silence gives an empty text
  instead of made-up words.
- **Speech output.** `--tts "text" -o out.wav`, and `POST /v1/audio/speech` as WAV, PCM or MP3, with a
  female or male voice. PCM and MP3 stream while they are generated (first audio after about 0.7 s),
  or as Server-Sent Events. MP3 uses the system's encoder, loaded at run time: Media Foundation on
  Windows, libmp3lame on Linux and macOS. Speech comes out 2.7 times faster than real time on a 4 GB
  laptop GPU (Vulkan), about real time on the CPU.
- **Voice conversations.** Spoken answers in English (`"modalities": ["text", "audio"]`, streamed);
  in Pandora, dictation with the microphone and a voice mode that reads the answers aloud; the new
  page **`/talk`** for hands-free conversations: Pandora listens, answers aloud and can be
  interrupted. Other languages go through transcription, a short answer and speech synthesis.
- The prompt follows the model's own chat template (no reasoning blocks unless reasoning is asked
  for), which fixes answers spoken twice in Italian conversations.

**Other changes**

- `--ppl` now tokenizes text as `llama-perplexity` does (Windows line endings, user-defined tokens),
  so perplexities compare directly.
- The landing page has a section on the speech server, with a Home Assistant example.

The text models are unchanged: greedy answers are identical to v0.6.2 on the CPU, OpenCL, Vulkan
and Direct3D 12, and speed is the same within measurement noise.

See the README for the full documentation and the measured results.
