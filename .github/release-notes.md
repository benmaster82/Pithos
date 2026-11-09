Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B and
LFM3-1.7B locally, with a terminal chat, the Pandora web chat, an OpenAI-compatible server and a
terminal agent. This release moves the prompt onto the GPU with OpenCL and makes the OpenCL
kernels portable across NVIDIA, AMD and Intel drivers.

## Downloads

| File | For |
|---|---|
| `pithos-0.6.0-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.6.0-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.6.0-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB), put it in a `models` folder next to the program, then run
for example `pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.5.2

- **The prompt on the GPU with OpenCL.** On a dedicated GPU with an int8 dot product in hardware,
  the prompt matmuls and the experts held in VRAM now run on the GPU, in tiles of 64 tokens, and
  the prompt batch grows to 256 tokens (unless `-b` is given). On the reference laptop
  (GTX 1650 4 GB), 512-token prompt:
  - LFM2.5-8B-A1B: 91 → about 270 tok/s (the default backend for this model);
  - LFM2.5-24B-A2B: about 52 → about 100 tok/s;
  - LFM3-1.7B with `--gpu opencl`: 102 → about 240 tok/s (Vulkan stays the default there).

  Generation speed and output quality do not change (perplexity matches the CPU within rounding).
  For the 8B model, OpenCL alone is now faster than `--gpu opencl+vulkan`.
- **Portable OpenCL kernels.** Pithos reads the driver's capabilities and builds the kernels for
  them: the int8 dot product from PTX on NVIDIA, from `cl_khr_integer_dot_product` where the driver
  accelerates it, emulated otherwise, and an f32 build if those fail. `--list-gpus` shows what each
  device offers.
- **Intel GPUs with OpenCL.** Dedicated matrix-vector kernels for Intel: LFM3 generation on an
  Intel UHD (Comet Lake) went from 3.6 to about 9.9 tok/s, close to the Vulkan build (10.2). On
  integrated GPUs the prompt stays on the CPU, which is faster there.
- New diagnostic variables for OpenCL (`PITHOS_GPU_DOT`, `PITHOS_CL_MIN`, `PITHOS_GPU_MOETILE`,
  `PITHOS_CL_CHECK` and others), listed in the README.

AMD GPUs and drivers with `cl_khr_integer_dot_product` have not been tested yet: if you have one,
the output of `pithos --list-gpus` and `pithos --bench` is very welcome in an issue.

See the README for the full documentation and the measured results.
