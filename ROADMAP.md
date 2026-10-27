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

## Done
- v0.4: dense layers in VRAM with fused kernels (14–18 tok/s on real text, up from 11–14).
- v0.3: agent mode with built-in tools and an MCP client.
- v0.2: OpenAI-compatible server, prompt reuse, web interface (now Pandora).
- v0.1: dependency-free C engine, expert cache on the GPU through OpenCL.
