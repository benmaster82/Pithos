Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B, LFM3-1.7B
and LFM3-Audio-2B locally, with a terminal chat, the Pandora web chat, an OpenAI-compatible server
and a terminal agent. This release adds the largest model so far: Qwen3.6-35B-A3B (experimental).

## Downloads

| File | For |
|---|---|
| `pithos-0.7.5-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.7.5-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.7.5-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB, LFM3-Audio-2B 1.3 GB with its audio projector, Qwen3.6-35B-A3B
20.5 GB), put it in a `models` folder next to the program, then run for example
`pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.7.0

**Qwen3.6-35B-A3B (experimental).** A 35-billion-parameter MoE: 256 experts of which 8, plus a
shared one, work on each token (about 3 billion active), with 30 Gated DeltaNet layers and 10 of
gated attention. It reasons before it answers and calls tools in XML, with the same chat, server,
Pandora and terminal agent as the LFM models. The file is
`Qwen3.6-35B-A3B-Q4_K_M.gguf` from
[ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (Apache 2.0);
on Linux, `bash build.sh --download-model=qwen3.6-35b` downloads and verifies it.

- **On the CPU** it runs on any computer with 16 GB of RAM: the 19.1 GiB file does not fit, so the
  experts stream from disk at first and the ones in use stay in RAM. Tokenizer and chat template
  match the model's own; perplexity 1.451 against llama.cpp's 1.467.
- **With Vulkan** (the default for it on a discrete NVIDIA GPU) every layer runs on the GPU and the
  routed experts on the CPU; the most used experts also live in the VRAM that is left
  (`PITHOS_VK_EXPERT_MB` sets how much), and the GPU computes them while the CPU computes the
  others. Prompts are cut in two halves, so the GPU works on one while the CPU computes the experts
  of the other.
- **Speed** on a laptop with a 6-core CPU, 16 GB and a 4 GB GPU (`--bench 512,128`, warm):

  | | Prompt | Generation |
  |---|---:|---:|
  | llama.cpp b12179, CPU only | 8.1 tok/s | 4.4 tok/s |
  | Pithos, CPU only | 13.6 tok/s | 7.3 tok/s |
  | llama.cpp, Vulkan, best of `-ncmoe 40` / `38` | 28.9 tok/s | 12.5 tok/s |
  | **Pithos, Vulkan** | **82 tok/s** | **22 tok/s** |

  Real conversations are slower on 16 GB, because varied text needs experts that are not in RAM:
  about 10–13 tok/s with the GPU.
- **Sampling** follows the model card when no option is given: temperature 1.0, top-k 20, top-p
  0.95, presence penalty 1.5 while it reasons; temperature 0.7 and top-p 0.8 with the reasoning off.

**Other changes**

- **Prompt reuse for agents.** The tool definitions (about 4,000 tokens for the agent and the web
  UI) are reused from the prompt cache even when the system text changes (the date, the folder), and
  even after a restart: a new session starts in seconds. Stored sessions of models with a large
  recurrent state are written with fewer checkpoints (350 MB instead of 1 GB and growing).
- **Tool calls in XML** (LFM3 and Qwen) keep a parameter as a string when the tool declares it so:
  `"0042"` or `"570"` no longer arrive as numbers.
- **`--presence-penalty`**, and the `presence_penalty` field in requests (OpenAI's meaning).
- **`PITHOS_VK_PROF=1`** prints the Vulkan GPU time of each step per token, at exit.

The LFM models are unchanged: greedy answers are identical to v0.7.0 on the CPU, Vulkan, Direct3D
12 and OpenCL for LFM3-1.7B, on the CPU, Vulkan and OpenCL for LFM2.5-8B, on the CPU for
LFM3-Audio-2B (transcription and speech included) and LFM2.5-24B; with OpenCL, whose expert cache
fills in the background, the 24B model's perplexity stays in v0.7.0's range.

See the README for the full documentation and the measured results.
