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
  latent noise is part of the output: two runs on the same input differ unless
  the draw is fixed. `--eps-std 0` makes it deterministic and `--seed` fixes the
  draw.
* Geometry comes from the checkpoint's own tensor names; `-m` is the whole
  choice.

Both backends reproduce the upstream PyTorch implementation's output to within
one level of 255 on a handful of values per image, none off by more; the CUDA
graph and the CPU graph agree to 1.2e-6 running the same fixed draw.

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
| default | samples at the checkpoint's own `eps_std` (0.9), a fresh draw each run |
| `--eps-std 0` | deterministic: no noise, byte-identical output run to run |
| `--eps-std <f>` | the temperature: below 0.9 is smoother, above is noisier |
| `--seed <n>` | fixes the latent draw, so input + seed reproduces exactly |

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
                      0.9; 0 is deterministic)
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
architecture, which is by [Zengyi-Qin](https://github.com/Zengyi-Qin/hcflow)
and MIT licensed. The converted `.safetensors` checkpoint is a format
conversion of the official `.pth` file, redistributed under the same MIT terms;
the original `.pth` file is not redistributed here.
