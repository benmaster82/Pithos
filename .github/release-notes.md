Pithos is a dependency-free C engine that runs Liquid AI's LFM2.5-24B-A2B, LFM2.5-8B-A1B and
LFM3-1.7B locally, with a terminal chat, the Pandora web chat, an OpenAI-compatible server and a
terminal agent. This is a bug-fix release for the OpenCL backend.

## Downloads

| File | For |
|---|---|
| `pithos-0.6.2-windows-x64.zip` | Windows 10 / 11, 64-bit (AVX2, AVX or scalar kernels chosen at run time) |
| `pithos-0.6.2-linux-x64.tar.gz` | Linux x86-64 with glibc 2.34 or newer (Ubuntu 22.04+, Debian 12+, RHEL / Rocky 9+, Fedora 35+...) |
| `pithos-0.6.2-linux-arm64.tar.gz` | Linux ARM64 (**experimental**): `pithos` for ARMv8.2 with dot product (Raspberry Pi 5, RK3566 / RK3588, Graviton 2+), `pithos-armv8` for older cores (Raspberry Pi 3 / 4) |
| `SHA256SUMS` | checksums of the archives |

The model is not included: download it as described in the README (LFM3-1.7B is 1.1 GB, the 8B
model 5.2 GB, the 24B model 14.5 GB), put it in a `models` folder next to the program, then run
for example `pithos --server --open`.

The Windows binary is not signed, so SmartScreen may warn about an unknown publisher. The ARM64
binaries passed the kernel tests and a short generation on GitHub's ARM64 runners (Neoverse); they
have not been tried on single-board computers yet. Reports are welcome.

## What's new since v0.6.1

- **OpenCL on integrated Intel GPUs under Linux.** The Linux Intel driver fails transfers to or from
  mapped host buffers on integrated GPUs with `CL_OUT_OF_RESOURCES` (-5)
  ([intel/compute-runtime#866](https://github.com/intel/compute-runtime/issues/866), fixed in the
  driver from 26.22 according to Intel). Pithos used such buffers for all its transfers, which is
  the likely cause of the `-5` reported in #3 on an Alder Lake iGPU (thanks again to
  @robertosantosx2). On integrated GPUs the transfers now go through ordinary memory; dedicated
  GPUs keep pinned memory. `PITHOS_GPU_PINNED=0` / `1` overrides the choice.
- README: GPU acceleration needs Q4_K_M files (a Q4_0 file runs on the CPU), and a note for
  OpenCL on Linux when the program exits right after loading the model: NVIDIA's compiler needs a
  writable `TMPDIR` (found by @robertosantosx2 too).

Speed is unchanged on the GPUs measured (a 4 GB NVIDIA laptop GPU, Intel UHD on Windows).

See the README for the full documentation and the measured results.
