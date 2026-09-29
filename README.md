# hcflow-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

HCFlow (ICCV 2021) conditional-flow x4 super-resolution in one self-contained
binary: give it a clean but low-resolution image and it reconstructs a 4x
version by sampling a learned conditional flow. No Python, PyTorch, ONNX
Runtime, or CUDA toolkit needed.

```sh
hcflow -m hcflow_x4.safetensors -i photo.png -o photo_4x.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and the
  CPU path otherwise, so one binary covers a machine with no NVIDIA driver at
  all; `--device cpu|gpu` overrides that choice.
* The model is STOCHASTIC. It is a conditional flow, not a regression, so the
  latent noise is part of the output: the default run draws it afresh each time,
  and two runs on the same input differ. `--seed <n>` repeats a draw, and
  `--eps-std 0` asks for the deterministic mean instead - no noise at all, so it
  is the same image whether or not a seed is given.
* Geometry comes from the checkpoint's own tensor names; `-m` is the whole
  choice.

Both backends reproduce the upstream PyTorch implementation's output to within
one level of 255 on a handful of values per image, none off by more; the CUDA
graph and the CPU graph agree to 1.2e-6 running the same fixed draw. The evidence
is built into the binary - `--cuda-selftest`, `--prim-test`, `--ng-test` and
`--seed-test`, the last of which checks the noise rules above.

## Download

Prebuilt binary and the converted checkpoint are attached to the
[release](https://github.com/jacobsparts/hcflow-rs/releases).

| asset | what it is |
|---|---|
| `hcflow-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present, the GPU path needs a compute capability 6.1+ GPU |
| `hcflow_x4.safetensors` | the x4 SR checkpoint, converted from the official `.pth` (88.7 MiB, 1372 tensors) |

```sh
chmod +x hcflow-linux-x86_64
./hcflow-linux-x86_64 -m hcflow_x4.safetensors -i photo.png -o photo_4x.png
```

## Models

One checkpoint, converted from the official `.pth` file; there is no model
choice to make. Unlike the family's regression engines, the run-to-run
variation is set by the sampling parameters, not by which file is loaded:

| setting | what it does |
|---|---|
| default | samples at the checkpoint's own `eps_std` (0.9), drawing the noise afresh each run |
| `--seed <n>` | draws that noise from the seed instead, so input + seed reproduces exactly |
| `--eps-std 0` | deterministic: no noise at all, so the output is the mean and is byte-identical run to run (and a seed is ignored) |
| `--eps-std <f>` | the temperature: below 0.9 is smoother, above is noisier |

The mean is the reproducible image a fixed input has; a drawn sample is the
model's characteristic texture. They are different pictures - on a detailed
256x256 crop the two modes differ by a mean 1.5/255 with 12% of pixels off by
more than 3 and a worst case of 144 - so a caller that needs one answer twice
wants `--eps-std 0`, and one that wants what the authors trained wants the
default.

The checkpoint is a x4 model: a 256x256 input becomes 1024x1024.

## Usage

```sh
hcflow -m hcflow_x4.safetensors -i photo.png -o photo_4x.png
hcflow -m hcflow_x4.safetensors -i photo.png -o photo_4x.png --eps-std 0
hcflow -m hcflow_x4.safetensors -i photo.png -o photo_4x.png --seed 7
hcflow -m hcflow_x4.safetensors -i photo.png -o photo_4x.png --device cpu
```

```
-m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
-i, --input <path>    input PNG, or - for stdin (default: stdin)
-o, --output <path>   output PNG, or - for stdout (default: stdout)
    --device <dev>    gpu or cpu (default: gpu when the CUDA driver can be
                      brought up, cpu otherwise)
    --eps-std <f>     sampling temperature (default: the checkpoint's own,
                      0.9; 0 is the mean, and ignores --seed)
    --seed <n>        seed for the latent noise (default: a fresh draw)
    --cpu             same as --device cpu
    --gpu             same as --device gpu, and refuses to fall back
-q, --quiet           no progress output
-h, --help            this text
-V, --version         print the version
```

## Large images

The whole image is resident on the device at once, so memory grows with the
input quadratically. The requirement is a calibrated envelope of measured peaks
rather than a guess, in MiB of input pixels:

| input | output | GPU peak | GPU time | CPU time |
|---|---|---|---|---|
| 64x64 | 256x256 | 140 MiB | 0.31 s | 2.4 s |
| 128x128 | 512x512 | 286 MiB | 0.71 s | 8.1 s |
| 256x256 | 1024x1024 | 875 MiB | 2.41 s | 33.9 s |
| 512x512 | 2048x2048 | 2836 MiB | 10.6 s | - |

A run whose estimate does not fit is refused before it allocates anything,
naming the input size, the estimate and the memory that is actually available:

```
error: not enough memory for a 2048x2048 input on the gpu backend
  this input needs about 56.78 GiB, from the measured peak at every size from
  64x64 to 512x512
  only 7.32 GiB is free on the device
  use a smaller input, or free device memory, or run on the cpu backend
```

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE). This is an independent reimplementation of the HCFlow
architecture, which is by
[JingyunLiang](https://github.com/JingyunLiang/HCFlow) and Apache-2.0 licensed.
The converted `.safetensors` checkpoint is a format conversion of the official
`SR_DF2K_X4_HCFlow++.pth` from the upstream Apache-2.0 release, redistributed
under the same terms; the original `.pth` file is not redistributed here.
