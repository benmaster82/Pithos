Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B, LFM3-1.7B
and LFM3-Audio-2B, and Qwen3.6-35B-A3B (experimental), locally, with a terminal chat, the Pandora web
chat, an OpenAI-compatible server and a terminal agent. This release gives the Xbox One app a voice:
you talk to the console through the controller's headset, and it answers out loud.

## Downloads

| File | For |
|---|---|
| `pithos-0.8.0-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.8.0-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.8.0-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB, LFM3-Audio-2B 1.3 GB with its audio projector, Qwen3.6-35B-A3B
20.5 GB), put it in a `models` folder next to the program, then run for example
`pithos --server --open`.

The Xbox app is not part of the downloads: it is signed with your own certificate and installed
from a Windows PC through the console's Device Portal, as the README's
[Xbox One](https://github.com/benmaster82/Pithos#xbox-one) section explains.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.7.5

**Voice on the Xbox One.** With LFM3-Audio-2B loaded, Pithos.X Server hears and speaks, entirely
on the console ([the story](https://benmaster82.github.io/Pithos/xbox.html)):

- **Push-to-talk** with a headset plugged into the controller's 3.5 mm jack: hold RT, speak,
  release. The question is transcribed in under a second and sent to the chat.
- **Spoken answers**: Y reads the last one or stops it, *Read aloud* speaks every answer, and RT
  talks over the console. The first words play after 0.5–0.8 seconds; playback waits until enough
  audio is queued to play the answer to the end without a gap.
- **Talk mode**: a full-screen assistant on a deep black background, with the Pithos mark, an aura
  and rings that follow the voice, neural-network layers with travelling pulses, the clock and
  subtitles. Answers are short and always spoken; LB / RB change the language (English, Italian,
  French, German, Spanish, Portuguese), ↑ / ↓ the reasoning level.
- **Hands-free** (X in Talk mode): the console listens for speech above the noise of the room,
  sends the question after a short pause, and raises its threshold while it speaks, so its own
  voice does not trigger it.
- **Reasoning off / low / medium / high** in the TV chat, off by default.
- Speech runs at about **0.87× real time** on the console in Game mode, up from 0.48×: the depth
  decoder runs on the GPU in one Direct3D 12 command list per pass (`PITHOS_SPEECH_DX=1`), and the
  audio codec decodes each frame on the CPU while the GPU makes the next one.

**Faster speech on the PC too.** The codec now overlaps the next step of the model whenever the
whole model runs on the GPU (Vulkan or Direct3D 12), and the depth decoder picks its top codes by
selection instead of sorting. LFM3-Audio-2B reading a sentence aloud on a laptop with a 6-core CPU
and a 4 GB GPU:

| | v0.7.5 | v0.8.0 |
|---|---:|---:|
| Vulkan | 2.8× real time | **3.4×** |
| Direct3D 12 | 2.3× real time | **3.1×** |
| CPU only | 1.3× real time | 1.4× |

The audio is byte-identical to v0.7.5's.

**Other changes**

- `PITHOS_SPEECH_PROFILE=1` also times `/v1/audio/speech` requests, and `=2` splits the depth
  decoder's time; `PITHOS_SPEECH_CODEC_THREADS` gives the codec threads of its own.
- `xbox/deploy.py` gained `get`, `touch` and `rm` for files in the app's folder; the app reads
  engine environment variables from `LocalState\pithos.env`.

The other models are unchanged: greedy answers are identical to v0.7.5 for LFM3-1.7B (CPU, Vulkan,
Direct3D 12, OpenCL), LFM2.5-8B (CPU, Vulkan), LFM2.5-24B (CPU), LFM3-Audio-2B (transcription and
speech) and Qwen3.6-35B-A3B (CPU and Vulkan, server and tool calls included), and their speed is the
same.

See the README for the full documentation and the measured results.
