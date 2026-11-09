Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B and
LFM3-1.7B locally, with a terminal chat, the Pandora web chat, an OpenAI-compatible server and a
terminal agent. This is a bug-fix release for the OpenCL backend.

## Downloads

| File | For |
|---|---|
| `pithos-0.6.1-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.6.1-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.6.1-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB), put it in a `models` folder next to the program, then run
for example `pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.6.0

- **A failing OpenCL GPU no longer produces garbage.** An OpenCL error was only printed: after a
  failed readback the generation went on from a buffer that was never written (thanks to
  @robertosantosx2 for finding it on an Intel iGPU under Linux, in #3, and for proposing the
  fail-closed handling). Now, as with Vulkan and D3D12, the GPU is not used again after the first
  failed call and the context restarts on the CPU. A one-shot answer (`-p`), `--raw`, `--ppl` and
  `--bench` stop with exit code 1, and the benchmark prints no numbers that would mix two devices;
  the interactive chat resets the conversation and goes on on the CPU.
- `PITHOS_CL_FAIL=N` simulates the failure at the Nth generated token, to test the fallback.

Speed and output are unchanged on working GPUs.

See the README for the full documentation and the measured results.
