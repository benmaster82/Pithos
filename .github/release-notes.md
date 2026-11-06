First release with prebuilt binaries. Pithos is a dependency-free C engine that runs Liquid AI's
LFM2.5-24B-A2B and LFM3-1.7B locally, with a terminal chat, the Pandora web chat, an
OpenAI-compatible server and a terminal agent.

## Downloads

| File | For |
|---|---|
| `pithos-0.5.0-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.5.0-linux-x64.tar.gz` | Linux x86-64 with glibc 2.35 or newer (Ubuntu 22.04+, Debian 12+, Fedora 36+...) |
| `pithos-0.5.0-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 24B
model 14.5 GB), put it in a `models` folder next to the program, then run for example
`pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.4

- **Vulkan backend** (Windows, Linux; experimental): the whole forward pass of dense models on the
  GPU. LFM3-1.7B on a GTX 1650: 328 tok/s of prompt, 82 of generation.
- **Direct3D 12 backend** (Windows, Xbox): the same for Direct3D 12 (330 / 69 on the same GPU).
- **One option for the GPU**: `--gpu auto|off|vulkan|d3d12|opencl|opencl+vulkan|opencl+d3d12`
  and `--cpu`. By default LFM3 runs on Vulkan on a discrete NVIDIA GPU and the 24B model on OpenCL;
  `--list-gpus` lists the devices of every backend, `--gpu-device N` picks one. The old options
  (`--no-gpu`, `--d3d12`, `--vulkan`) still work.
- **LFM3-1.7B support**: sliding-window and gated attention, the LFM3 chat template and tool calls.
- **Reasoning levels**: `--reasoning high|medium|low|off` or a token budget, also per request.
- **AVX kernels** for x86-64 CPUs without AVX2.
- **Xbox One app** (built from source, see the README): the server on a console in Developer Mode.
- Fixes: after a GPU failure the engine continues on the CPU from an empty context instead of
  returning blank answers; the server reports the Vulkan backend.

See the README for the full documentation and the measured results.
