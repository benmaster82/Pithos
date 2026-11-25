Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B, LFM3-1.7B
and LFM3-Audio-2B, and Qwen3.6-35B-A3B and Inkling 2 Mini (experimental), locally, with a terminal
chat, the Pandora web chat, an OpenAI-compatible server and a terminal agent. This release adds the
smallest Inkling 2 Mini file, which fits in 12 GB of RAM.

## Downloads

| File | For |
|---|---|
| `pithos-0.8.6-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.8.6-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.8.6-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB, LFM3-Audio-2B 1.3 GB with its audio projector, Qwen3.6-35B-A3B
20.5 GB, Inkling 2 Mini 11.7 GB, or 8.0 GB for its smallest file), put it in a `models` folder next
to the program, then run for example `pithos --server --open`. Inkling's files come from Hugging
Face named `.gguf_file`: rename them to `.gguf` (on Linux, `bash build.sh
--download-model=inkling-2-mini` or `--download-model=inkling-2-mini-iq1m` does it for you).

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.8.5

**Inkling 2 Mini in 12 GB of RAM (experimental).** Pithos now runs Unsloth's smallest file,
UD-IQ1_M (7.5 GiB, 1–2 bits per weight), next to the 3-bit UD-IQ3_XXS (10.9 GiB), which stays the
default. It is a little faster, but further from the original model: against the logits of the
4-bit Q4_K_M file its KL divergence is 0.246 and its most likely token the same 85.6% of the time
(UD-IQ3_XXS: 0.049 and 92.9%). Short answers, code and Italian came out right in our tests, with
reasoning about as long as with the 3-bit file. On a laptop with a 6-core CPU, 16 GB of RAM and a
4 GB GPU, a 249-token prompt then 128 tokens:

| Inkling 2 Mini | Prompt | Generation |
|---|---:|---:|
| llama.cpp b12670, CUDA, experts on the CPU, UD-IQ1_M | 29.6 tok/s | 7.0 tok/s |
| Pithos, CPU only, UD-IQ1_M | 11 tok/s | 5.3 tok/s |
| Pithos, Vulkan, UD-IQ3_XXS | 17–19 tok/s | 9.6–9.7 tok/s |
| **Pithos, Vulkan, UD-IQ1_M** | **18–20 tok/s** | **10.0 tok/s** |

Checked against llama.cpp on the same file and tokens: KL divergence 0.011, the same most likely
token 96.9% of the time, as close as with the 3-bit file.

**New weight formats.** IQ1_S and Q2_0 (a 2-bit format in blocks of 64 values), the formats of
Unsloth's smallest Inkling files, with AVX2 kernels; their dequantization matches llama.cpp's bit
for bit. On Vulkan their experts go to VRAM too, Q2_0 with a kernel of its own. UD-IQ2_XXS loads as
well, but it is slower than UD-IQ3_XXS here (9.2 tok/s with Vulkan), so it is not in the catalog.

The other models are unchanged: greedy answers are identical to v0.8.5 for LFM3-1.7B (CPU, Vulkan,
Direct3D 12, OpenCL), LFM2.5-8B (CPU, Vulkan), LFM2.5-24B (CPU), LFM3-Audio-2B (transcription and
speech on the CPU, Direct3D 12 and Vulkan), with the server and tool calls of the LFM models, and
Qwen3.6-35B-A3B (CPU and Vulkan), and their speed is the same.

See the README for the full documentation and the measured results.
