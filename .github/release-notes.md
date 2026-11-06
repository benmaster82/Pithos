Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B and LFM3-1.7B locally,
with a terminal chat, the Pandora web chat, an OpenAI-compatible server and a terminal agent.
This release makes the Vulkan backend faster on Intel GPUs; everything else works as in v0.5.0.

## Downloads

| File | For |
|---|---|
| `pithos-0.5.1-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.5.1-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.5.1-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 24B
model 14.5 GB), put it in a `models` folder next to the program, then run for example
`pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.5.0

- **Vulkan on Intel GPUs**: Intel devices get their own build of the four matmul kernels
  (other GPUs and Direct3D 12 run exactly the same code as before). LFM3-1.7B on an Intel UHD
  Graphics (i5-10500H): prompt 35 -> 53 tokens per second, generation 8.4 -> 10. Same results
  (perplexity unchanged); `PITHOS_VK_INTEL=0` switches back to the common kernels.
- On that UHD the CPU is still faster (85 / 21), so `--gpu auto` keeps leaving integrated GPUs
  off; use `--gpu vulkan --gpu-device N` (see `--list-gpus`) to run on one.

See the README for the full documentation and the measured results.
