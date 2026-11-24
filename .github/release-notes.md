Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B, LFM3-1.7B
and LFM3-Audio-2B, and Qwen3.6-35B-A3B and Inkling 2 Mini (experimental), locally, with a terminal
chat, the Pandora web chat, an OpenAI-compatible server and a terminal agent. This release adds
Inkling 2 Mini, Thinking Machines' looped mixture of experts, and the IQ formats of the smaller GGUF
files.

## Downloads

| File | For |
|---|---|
| `pithos-0.8.5-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.8.5-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.8.5-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB, LFM3-Audio-2B 1.3 GB with its audio projector, Qwen3.6-35B-A3B
20.5 GB, Inkling 2 Mini 11.7 GB), put it in a `models` folder next to the program, then run for
example `pithos --server --open`. Inkling's file comes from Hugging Face named `.gguf_file`: rename
it to `.gguf` (on Linux, `bash build.sh --download-model=inkling-2-mini` does it for you).

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.8.0

**Inkling 2 Mini (experimental).** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
is a looped transformer: the middle 12 of its 24 layers run three times with the same weights, so
each token goes through 48 layers; its feed-forward layers are mixtures of 128 experts, 8 per token
(26.4 billion parameters, 3.4 billion active per pass). Attention has no rotary positions but a
learned relative bias, and the later passes of the loop get messages from the earlier ones. It
reasons before it answers, with the model's own thinking-effort levels (`--reasoning`, or
`reasoning_effort` in the API, also inside `chat_template_kwargs`), and calls tools in JSON, in the
chat, the server, Pandora and the terminal agent. Images are not supported yet.

Pithos takes Unsloth's 3-bit file (UD-IQ3_XXS, 10.9 GiB), which fits in 16 GB of RAM; the 4-bit one
(UD-Q4_K_M) runs too. With Vulkan, the default on a discrete NVIDIA GPU, every layer runs on the
GPU and the routed experts on the CPU, the most used of them also in the VRAM that is left (about
450 on a 4 GB GPU). On a laptop with a 6-core CPU, 16 GB of RAM and a 4 GB GPU, a 249-token prompt
then 128 tokens:

| Inkling 2 Mini | Prompt | Generation |
|---|---:|---:|
| llama.cpp b12670, CUDA, experts on the CPU, UD-IQ3_XXS | 5.5–6.1 tok/s | 4.0–4.6 tok/s |
| Pithos, CPU only, UD-IQ3_XXS | 11 tok/s | 4.6–5.5 tok/s |
| Pithos, Vulkan, UD-Q4_K_M | 6.0 tok/s | 8.1–8.4 tok/s |
| **Pithos, Vulkan, UD-IQ3_XXS** | **17–19 tok/s** | **9.6–9.7 tok/s** |

Checked against llama.cpp on the same files and tokens: the same most likely token 97% of the time;
the tokenizer gives the same ids as Hugging Face's and the chat template the same text as the
model's own.

**New weight formats.** NVFP4 (the 4-bit format Inkling was trained in), Q5_0 and the IQ codebook
formats IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS, with AVX2 kernels; their
dequantization matches llama.cpp's bit for bit. On Vulkan, the IQ2 and IQ3_XXS experts go to VRAM
in a compact layout of their own.

**Other changes**

- The landing page groups the models by the memory they need (4, 8 or 16 GB).
- Xbox: `xbox/deploy.py use MODEL` switches the console's model in one step, and the saved
  conversations stay within 256 MB on the console (those of a model that is not loaded wait on
  the PC).

The other models are unchanged: greedy answers are identical to v0.8.0 for LFM3-1.7B (CPU, Vulkan,
Direct3D 12, OpenCL), LFM2.5-8B (CPU, Vulkan), LFM2.5-24B (CPU), LFM3-Audio-2B (transcription and
speech), with the server and tool calls of the LFM models, and Qwen3.6-35B-A3B (CPU and Vulkan),
and their speed is the same.

See the README for the full documentation and the measured results.
