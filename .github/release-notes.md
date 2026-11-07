Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B and LFM3-1.7B locally,
with a terminal chat, the Pandora web chat, an OpenAI-compatible server and a terminal agent.
This release fixes `--bench` with long prompts; everything else works as in v0.5.1.

## Downloads

| File | For |
|---|---|
| `pithos-0.5.2-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.5.2-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.5.2-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 24B
model 14.5 GB), put it in a `models` folder next to the program, then run for example
`pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.5.1

- **`--bench` with prompts longer than the context**: the prompt pass was skipped and reported as
  0.00 s (thanks to @robertosantosx2 for spotting it in #2). The context now grows to fit the
  benchmark; with an explicit `--ctx` that is too small, `--bench` stops with an error instead.
  Chat, server and agent were not affected.

See the README for the full documentation and the measured results.
