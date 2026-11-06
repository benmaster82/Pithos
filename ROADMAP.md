# Pithos roadmap

Status as of 2026-10-24 (v0.4). Reference hardware: i5-10500H, 16 GB, GTX 1650 Max-Q 4 GB.

A generated token currently costs about 55–70 ms:

| Item | ms/token |
|---|---:|
| Non-resident experts computed on the CPU | 30–45 |
| Waiting for the GPU | ~10 |
| Command queuing and combine | ~8 |
| Logits | ~3.5 |

Sections are ordered by expected impact.

## 1. Experts on the CPU (current bottleneck)
The model (13.5 GB) does not fit in the free RAM (9–10 GB), so part of the experts is read from disk
during generation.

The experts on the CPU are bound by RAM bandwidth (~660 MB of weights per token at ~20 GB/s), so
time scales with the bytes read.

- [ ] **Experts in a more compact format** (3 bits or less; Q4_0 does not help, it is as large as Q4_K).
  - **Best candidate:** [APEX I-Compact](https://huggingface.co/mudler/LFM2.5-24B-A2B-APEX-GGUF), a
    mixed quantization of this model made from BF16 with an imatrix.
    - Size: 10.4 GiB instead of 13.5.
    - Quality: perplexity +1.6% (16.20 vs 15.94), HellaSwag unchanged within noise.
  - **Check first:** read the file metadata to see which formats it uses. They are probably
    IQ4_XS/IQ3, which Pithos does not support today.
  - **Not re-quantizing ourselves:** compressing again from Q4_K would give worse results (errors
    add up, and there is no imatrix).
  - **Dynamic variant later:** cold experts compact and hot ones at 4 bits, in a side file. Only once
    the static file holds up.
  - **Expected gain:** from 14–18 to ~17–21 tok/s, and more stable.
- [ ] **Predictor-driven VRAM prefetch**: upload to the GPU the experts predicted for the next layer.
  Modest gain: 1–2 uploads fit per layer, at ~0.27 ms per expert.
- [ ] **More expert slots in VRAM** by freeing memory (int8 KV cache, see section 4). Marginal at an
  8k context (~15 experts, +0.6%), useful only with long contexts.

## 2. Speculative decoding without a draft model (prompt lookup)
- [ ] Propose the next tokens by looking up n-grams already in the context, and verify them in a
  single batched pass (the experts are shared between the tokens of the batch).
  - Expected: +30–80% on repetitive text, such as the agent's file names, code and tool results.
  - Little gain on free text.

## 3. Prefill on the GPU
- [ ] Move the dense layers of prompt processing to the GPU as well. Today they run batched on the
  CPU: 36–45 tok/s warm, much less cold. This cuts the wait for the first answer in agent mode,
  where the system prompt with the tools is long.

## 4. Features
- [ ] **Agent cold start**: precompute the system prompt with the tools at start-up.
- [ ] **Constrained JSON output**: a grammar that guarantees well-formed tool calls.
- [ ] **Longer context / int8 KV cache**: less VRAM and RAM per context token.
- [ ] **Parallel requests in the server** (today one at a time, the others wait in a queue).
- [ ] **Testing on other hardware**: NEON kernels on ARM64, the f32 GPU path on AMD and Intel GPUs.

## 5. Polish
- [ ] Fuse the MoE combine into the first kernel of the next layer (ping-pong residual stream):
  ~1 ms per token.

## 6. Direct3D 12 backend and Xbox (branch `xbox`)
Measured on the Xbox One S in Game mode (LFM3-1.7B Q4_K_M): prompt 146 tok/s at 125 tokens and
121 at 1000, generation 29.5 tok/s (26 after 1000 tokens of context). The GPU reads ~49 GB/s, and
generation uses about 30 of them.

- [ ] **Faster batched matmul kernels** (`mm_q4_K` / `mm_q6_K`): almost all of a prompt chunk's time
  goes there. Try integer dot products on the Q8_K activations, as `mv_*` already does, and larger
  tiles. A bigger batch (`-b 256`) gains only ~4% today.
- [ ] **Attention over long prompts**: `attn_chunk` runs one group per head and token, reading the
  K/V rows again for every token. Tiles of several queries sharing the loads (flash-attention style)
  should remove the slowdown from 146 to 121 tok/s between 125 and 1000 tokens.
- [ ] **Faster generation**: about 1.5x of bandwidth headroom left. Fewer barriers and dispatches
  per token (for example, quantization fused into the norm kernels), and a look at the `mv_*`
  memory access pattern.
- [ ] **Logits of every token on the GPU**, so that `--ppl` and `ctx_forward_all` use the fast path too.
- [ ] **`--gpu auto` on more GPUs**: it picks Vulkan for dense models on discrete NVIDIA GPUs only
  (done, see below). AMD (Vulkan or Direct3D 12) once measured; Direct3D 12 where Vulkan is missing.
- [ ] **Experts on Direct3D 12**: the backend handles models without experts only. The 24B model
  does not fit in the Xbox's 5 GB, but it matters for PCs.
- [ ] **Xbox app type**: every reinstall resets the app to "App", and the user must set "Game"
  again in Dev Home before Direct3D 12 is available. Look for a way to declare it in the package.
- [ ] **Merge `xbox` into `main`** after review: shared-code changes are small and measured neutral
  on Windows and Linux.

## 7. Vulkan backend (`--gpu vulkan`, experimental)
The Direct3D 12 kernels compiled to SPIR-V; same results on the same GPU. LFM3-1.7B on the GTX 1650:
prompt 328 tok/s, generation 82 (Direct3D 12: 336 / 70).

- [ ] **Testing on AMD and on real Linux GPUs**: so far NVIDIA and Intel UHD on Windows, and only
  Mesa's software driver (llvmpipe) on Linux.
- [ ] **Batched matmuls on Intel**: `mm_*` compile to SIMD8 with register spills there (35 tok/s of
  prompt against 42 with Direct3D 12). Integrated Intel GPUs stay slower than the CPU anyway.
- [ ] **KV cache over the buffer limit**: split it like the large matrices, so that small-buffer
  devices (llvmpipe: 128 MB) keep the whole forward pass at the default context.
- [x] **Vulkan by default where it is fastest**: `--gpu auto` on discrete NVIDIA GPUs, for dense
  models.

## Done
- Branch `gpu-cli`: one option for the GPU, `--gpu auto|off|vulkan|d3d12|opencl|opencl+vulkan|opencl+d3d12`
  (and `--cpu`), with `--gpu-device N` for every backend and `--list-gpus` listing all three; by
  default LFM3 runs on Vulkan on a discrete NVIDIA GPU (82 tok/s instead of 56). The old options
  keep their meaning.
- Branch `vulkan`: Vulkan backend (`--vulkan`, `--vulkan-device N`) on Windows and Linux, loaded at
  runtime, with the whole forward pass of dense models and the matmuls for MoE models.
- Branch `xbox` (not released yet): UWP app for the Xbox One in Dev Mode with a status and chat
  screen on the TV, AVX kernels for CPUs without AVX2 (Jaguar, Ivy Bridge), and a Direct3D 12
  backend (`--d3d12`) running the whole forward pass on the GPU. On the Xbox: prompt 13.9 → 146
  tok/s, generation 5.8 → 29.5 tok/s.
- v0.4: dense layers in VRAM with fused kernels (14–18 tok/s on real text, up from 11–14).
- v0.3: agent mode with built-in tools and an MCP client.
- v0.2: OpenAI-compatible server, prompt reuse, web interface (now Pandora).
- v0.1: dependency-free C engine, expert cache on the GPU through OpenCL.
