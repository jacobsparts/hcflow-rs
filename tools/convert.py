#!/usr/bin/env python3
"""Convert an HCFlow x4 SR checkpoint (.pth) to a lightgpu-family .safetensors.

The Rust engine is an INFERENCE-ONLY port, so this converter performs the
algebra that inference does not need to repeat, and proves each fold with
numbers before writing the file.

FOLDS
-----
1. ActNorm2d.  At inference an ActNorm is ``y = (x + bias) * exp(logs)`` and its
   reverse is ``y = x * exp(-logs) - bias``: both are a per-channel affine with
   CONSTANT vectors.  Where an ActNorm's output feeds a convolution and nothing
   else reads it, it folds into that convolution:
   ``W' = W * exp(logs)[:,None]``, ``b' = (b + bias) * exp(logs)``.  The folds
   this converter performs:

     * the ActNorm on a coupling net's conv1 / conv2 output (conv1 and conv2
       have no bias of their own, so the bias IS the actnorm's);
     * the Conv2dZeros' ``logscale_factor`` scale on conv3
       (``W *= exp(3*logs)[:,None]``), which is the same shape of fold.

   What is NOT folded, and why:
     * a FlowStep's own ActNorm: its output is read by the 1x1 permutation,
       which would have to be rewritten too.  The engine applies the reverse
       form ``x*exp(-logs) - bias`` directly, which is 2 mul-adds per element -
       cheaper than the fold would be, since the fold would need two passes
       over a buffer that already exists;
     * an ActNorm that is the LAST op of a subgraph whose output feeds several
       consumers (the conditional flow's trunk_conv1, and the FCN's conv3 in
       the *forward* direction).  The engine scales in place there.

2. InvertibleConv1x1.  The reverse direction uses ``torch.inverse(W)``, computed
   in float64 at every call in the reference.  It is a constant, so it is
   computed ONCE here, in float64, exactly as the reference does, and stored as
   ``*.w_rev``.  The forward weight is stored as ``*.w``.

3. The conditional flow's output layer ``f`` (Conv2dZeros) keeps its scale
   FOLDED IN, so the engine gets ``mean`` and ``logs`` by slicing its output.

LAYOUT
------
Tensors are stored in the reference's own layout, ``[c_out][c_in][ky][kx]`` for
convolutions and ``[c_out][c_in]`` for the 1x1s, so no transposition happens
here and the Rust side's shapes read the same as PyTorch's.

Usage:
    python3 tools/convert.py SR_DF2K_X4_HCFlow++.pth hcflow_x4.safetensors
    python3 tools/convert.py --check-only in.pth
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

ALIGN = 8
FORMAT = "hcflow-safetensors-v1"


# --------------------------------------------------------------------------
# reference arithmetic, re-implemented here so the folds can be CHECKED
# --------------------------------------------------------------------------

def ref_actnorm(x, bias, logs):
    """y = (x + bias) * exp(logs), per channel, NCHW fp32."""
    b = bias.reshape(1, -1, 1, 1)
    s = np.exp(logs.reshape(1, -1, 1, 1))
    return ((x + b) * s).astype(np.float32)


def ref_actnorm_rev(y, bias, logs):
    b = bias.reshape(1, -1, 1, 1)
    s = np.exp(-logs.reshape(1, -1, 1, 1))
    return (y * s - b).astype(np.float32)


def ref_conv2d(x, w, b, stride=1, pad=0):
    """F.conv2d over NCHW fp32, implemented as an explicit im2col + GEMM so it
    is obviously the same op and not a library that might differ."""
    n, c, h, wd = x.shape
    co, ci, kh, kw = w.shape
    assert c == ci and kh == kw
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c * kh * kw, oh * ow), dtype=np.float32)
    idx = 0
    for dy in range(kh):
        for dx in range(kw):
            patch = xp[:, :, dy:dy + stride * oh:stride, dx:dx + stride * ow:stride]
            cols[:, idx * c:(idx + 1) * c, :] = patch.reshape(n, c, oh * ow)
            idx += 1
    wm = w.reshape(co, c * kh * kw)
    out = np.einsum("ok,nkp->nop", wm, cols, optimize=True)
    if b is not None:
        out = out + b.reshape(1, -1, 1)
    return out.reshape(n, co, oh, ow).astype(np.float32)


def ref_conv1x1(x, w, b=None):
    """F.conv2d with a 1x1 kernel - no padding, no stride games."""
    n, c, h, wd = x.shape
    co = w.shape[0]
    # tensordot leaves the contracted operand's remaining axes (here just `o`,
    # which has no free axis left) in front of x's, and a length-1 o would be
    # silently broadcast against the batch if it did not simply fail. Be
    # explicit: contract over the channel axis and hand the result x's own
    # axis order back.
    w = w.reshape(w.shape[0], w.shape[1])       # the checkpoint stores 1x1s as
                                                # (oc, ic, 1, 1), like F.conv2d
    y = np.einsum("oc,nchw->nohw", w, x)
    if b is not None:
        y = y + b.reshape(1, -1, 1, 1)
    assert y.shape == (n, co, h, wd), y.shape
    return y.astype(np.float32)


def fold_scale(w, b, logs):
    """W' = W * exp(logs)[:,None], b' = b * exp(logs).  b may be None."""
    s = np.exp(logs.astype(np.float32)).reshape(-1, 1, 1, 1)
    w2 = (w.astype(np.float32) * s).astype(np.float32)
    b2 = None if b is None else (b.astype(np.float32) * s.reshape(-1)).astype(np.float32)
    return w2, b2


def fold_actnorm(w, b, an_bias, an_logs):
    """Fold y = (x + an_bias) * exp(an_logs) that PRECEDES a conv into the conv.

    An ActNorm feeding a convolution can be folded into the convolution's
    weights when the ActNorm sits AFTER it (``scale_out``) - which is the common
    case here.  This helper is the other direction, used nowhere in HCFlow but
    kept for symmetry tests.
    """
    raise NotImplementedError


def fold_actnorm_out(w, b, an_bias, an_logs):
    """Fold y = (conv(x) + an_bias) * exp(an_logs) into conv's weights+bias.

    ``b`` may be None, in which case the actnorm's bias becomes the conv's.
    """
    s = np.exp(an_logs.astype(np.float32)).reshape(-1, 1, 1, 1)
    w2 = (w.astype(np.float32) * s).astype(np.float32)
    base = np.zeros(w.shape[0], dtype=np.float32) if b is None else b.astype(np.float32)
    b2 = ((base + an_bias.astype(np.float32).reshape(-1)) * s.reshape(-1)).astype(np.float32)
    return w2, b2


# --------------------------------------------------------------------------
# checkpoint reading
# --------------------------------------------------------------------------

class Ck:
    """A checkpoint with the accessors the fold code reads naturally."""

    def __init__(self, path):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(obj, "state_dict"):
            obj = obj.state_dict()
        if isinstance(obj, dict) and "params" in obj and isinstance(obj["params"], dict):
            obj = obj["params"]
        if not isinstance(obj, dict) or not all(hasattr(v, "shape") for v in obj.values()):
            raise SystemExit(f"{path}: not a state dict ({type(obj)})")
        self.sd = {k: v.detach().cpu().float().numpy() for k, v in obj.items()}

    def has(self, name):
        return name in self.sd

    def raw(self, name):
        if name not in self.sd:
            raise SystemExit(f"checkpoint has no `{name}`")
        return self.sd[name]

    def get(self, name):
        return self.raw(name).astype(np.float32).copy()


# --------------------------------------------------------------------------
# the folds, one per graph position
# --------------------------------------------------------------------------

def flow_step(ck, base, fold_actnorm_in):
    """One FlowStep, reverse direction.

    Returns a dict of tensors:
      an_bias/an_logs   - the step's ActNorm, used in its REVERSE form
      w_rev             - the inverse 1x1, ready as an F.conv2d weight
      c1w/c1b           - the coupling net's conv1 with its ActNorm folded in
      c2w/c2b           - conv2 likewise
      c3w/c3b           - Conv2dZeros with its logscale_factor folded in
    ``fold_actnorm_in`` folds the step's own ActNorm into the incoming 1x1
    (which is only legal when the caller re-reads the actnorm off the folded
    matrix); it is off in the shipped path.
    """
    out = {}
    an_bias = ck.get(f"{base}.actnorm.bias").reshape(-1)
    an_logs = ck.get(f"{base}.actnorm.logs").reshape(-1)
    # The engine only ever runs the ActNorm in REVERSE, which is
    # y = x * exp(-logs) - bias. Rather than make every backend compute an
    # exponential and a negation per step - and rather than store a pair that
    # has to be remembered as "the reverse one" - the reversal is done here, in
    # float32, exactly as the reference's `ref_actnorm_rev` does it. The two
    # tensors the engine reads are therefore already the scale/shift of the
    # affine it applies, with no direction left to get wrong.
    out["an_scale"] = np.exp(-an_logs.astype(np.float32)).astype(np.float32)
    out["an_bias"] = (-an_bias.astype(np.float32)).astype(np.float32)
    w = ck.get(f"{base}.permute.weight")
    out["w"] = w
    out["w_rev"] = torch.inverse(torch.from_numpy(w).double()).float().numpy().astype(np.float32)

    f = f"{base}.affine.f"
    c1w = ck.get(f"{f}.conv1.weight")
    c1b = ck.get(f"{f}.conv1.actnorm.bias").reshape(-1)
    c1l = ck.get(f"{f}.conv1.actnorm.logs").reshape(-1)
    c1w, c1b = fold_actnorm_out(c1w, None, c1b, c1l)
    out["c1w"], out["c1b"] = c1w, c1b

    c2w = ck.get(f"{f}.conv2.weight")
    c2b = ck.get(f"{f}.conv2.actnorm.bias").reshape(-1)
    c2l = ck.get(f"{f}.conv2.actnorm.logs").reshape(-1)
    c2w, c2b = fold_actnorm_out(c2w, None, c2b, c2l)
    out["c2w"], out["c2b"] = c2w, c2b

    c3w = ck.get(f"{f}.conv3.weight")
    c3b = ck.get(f"{f}.conv3.bias").reshape(-1)
    c3l = ck.get(f"{f}.conv3.logs").reshape(-1)
    # Conv2dZeros: y = conv(x) * exp(logs * logscale_factor)
    c3w, c3b = fold_scale(c3w, c3b, c3l * 3.0)
    out["c3w"], out["c3b"] = c3w, c3b
    return out


def coupling_reference(ck, base, z, u, c1w, c1b, c2w, c2b, c3w, c3b):
    """The reference coupling's REVERSE direction, computed with the folded
    weights - so the check compares the folded pipeline against the unfolded
    one expressed through the same helpers."""
    raise NotImplementedError


def coupling_fwd_reference(ck, base, z, u):
    """The REFERENCE coupling in the FORWARD direction, straight from the
    checkpoint - no folds.  Used by --check to build an input for the reverse
    direction rather than inventing one (the reverse map's image is not the
    whole of R^n)."""
    f = f"{base}.affine.f"
    half = z.shape[1] // 2
    z1, z2 = z[:, :half], z[:, half:]
    h_in = z1 if u is None else np.concatenate([z1, u], axis=1)
    h = ref_conv2d(h_in, ck.get(f"{f}.conv1.weight"), None, pad=1)
    h = ref_actnorm(h, ck.get(f"{f}.conv1.actnorm.bias").reshape(-1),
                    ck.get(f"{f}.conv1.actnorm.logs").reshape(-1))
    h = np.maximum(h, 0.0)
    h = ref_conv1x1(h, ck.get(f"{f}.conv2.weight"))
    h = ref_actnorm(h, ck.get(f"{f}.conv2.actnorm.bias").reshape(-1),
                    ck.get(f"{f}.conv2.actnorm.logs").reshape(-1))
    h = np.maximum(h, 0.0)
    h = ref_conv2d(h, ck.get(f"{f}.conv3.weight"), ck.get(f"{f}.conv3.bias").reshape(-1), pad=1)
    h = h * np.exp(ck.get(f"{f}.conv3.logs").reshape(1, -1, 1, 1) * 3.0)
    shift, scale = h[:, 0::2], h[:, 1::2]
    logscale = 0.318 * np.arctan(2.0 * scale)
    z2n = (z2 + shift) * np.exp(logscale)
    return np.concatenate([z1, z2n], axis=1).astype(np.float32)


def coupling_rev_folded(step, z, u):
    """The coupling's REVERSE direction using the FOLDED conv weights - the
    path the Rust engine takes."""
    half = z.shape[1] // 2
    z1, z2 = z[:, :half], z[:, half:]
    h_in = z1 if u is None else np.concatenate([z1, u], axis=1)
    h = ref_conv2d(h_in, step["c1w"], step["c1b"], pad=1)
    h = np.maximum(h, 0.0)
    h = ref_conv1x1(h, step["c2w"], step["c2b"])
    h = np.maximum(h, 0.0)
    h = ref_conv2d(h, step["c3w"], step["c3b"], pad=1)
    shift, scale = h[:, 0::2], h[:, 1::2]
    logscale = 0.318 * np.arctan(2.0 * scale)
    z2n = z2 * np.exp(-logscale) - shift
    return np.concatenate([z1, z2n], axis=1).astype(np.float32)


def flow_step_rev_folded(step, z, u=None):
    """The full reverse of one FlowStep, folded: coupling -> inv 1x1 -> actnorm.

    ``u`` is the conditional feature for a conditional step; it is None for the
    unconditional ones, whose coupling has no cond_channels at all."""
    z = coupling_rev_folded(step, z, u)
    z = ref_conv1x1(z, step["w_rev"])
    # an_scale/an_bias are already the REVERSE affine's constants.
    z = z * step["an_scale"].reshape(1, -1, 1, 1) + step["an_bias"].reshape(1, -1, 1, 1)
    return z.astype(np.float32)


def flow_step_fwd_reference(ck, base, z, u=None):
    """The full FORWARD pass of one FlowStep, unfolded."""
    z = ref_actnorm(z, ck.get(f"{base}.actnorm.bias").reshape(-1),
                    ck.get(f"{base}.actnorm.logs").reshape(-1))
    z = ref_conv1x1(z, ck.get(f"{base}.permute.weight"))
    z = coupling_fwd_reference(ck, base, z, u)
    return z.astype(np.float32)


# --------------------------------------------------------------------------
# the emitted graph
# --------------------------------------------------------------------------

def rrdb_block(ck, base):
    """One RRDB trunk: its stack of ``RRDB`` blocks.

    ``RRDB`` is three ``ResidualDenseBlock``s in series with a single outer
    residual (``out * 0.2 + x``); each RDB is five dense 3x3 convolutions with
    ``lrelu(0.2)`` after all but the last, the inputs of each being the
    concatenation of the block's input and every previous dense output.  There
    is no actnorm and no inner residual, so this is a pure conv/relu stack and
    needs no folding.

    The block count is read from the checkpoint rather than taken from the
    config (``RRDB_nb = [7, 7]``): the shipped model has 7 blocks per trunk, and
    a converter that trusted a config number would emit a graph that loads
    cleanly and produces the wrong picture.
    """
    depth = len(base.split("."))
    names = sorted({k.split(".")[depth] for k in ck.sd if k.startswith(f"{base}.")},
                   key=int)
    blocks = []
    for n in names:
        rdb_names = sorted({k.split(".")[depth + 1] for k in ck.sd
                            if k.startswith(f"{base}.{n}.RDB")},
                           key=lambda s: int(s[3:]))
        rdbs = []
        for rdb in rdb_names:
            convs = {}
            for c in range(1, 6):
                convs[f"w{c}"] = ck.get(f"{base}.{n}.{rdb}.conv{c}.weight")
                convs[f"b{c}"] = ck.get(f"{base}.{n}.{rdb}.conv{c}.bias").reshape(-1)
            rdbs.append(convs)
        blocks.append(rdbs)
    assert blocks and all(len(b) == 3 for b in blocks), "unexpected RRDB layout"
    return {"blocks": blocks}


def to_tensors(t):
    """Flatten a nested dict/list into {name: ndarray}."""
    out = {}

    def walk(prefix, v):
        if isinstance(v, dict):
            for k, vv in v.items():
                walk(f"{prefix}.{k}" if prefix else k, vv)
        elif isinstance(v, (list, tuple)):
            for i, vv in enumerate(v):
                walk(f"{prefix}.{i}", vv)
        else:
            out[prefix] = np.ascontiguousarray(v, dtype=np.float32)
    walk("", t)
    return out


def build(ck, level):
    """One level's parameters, indexed the way the WALK uses them.

    ``level`` is the ENGINE's level index.  The reference builds its levels in
    the opposite order from the one inference visits them: ``FlowNet_SR_x4``
    appends the 12-channel block first and the 24-channel block second, while
    ``reverse_flow`` iterates ``reversed(self.layers)`` and therefore starts
    with the second block's Split.

    Geometry of the walk (HR = 4 * LR, fp32, L = 2):

      walk level 0   checkpoint block ``flow.layers.16..28`` + ``level1_condFlow``
                     z enters as 3 channels at LR - the lr image itself.  The
                     Split consumes those 3 and cats on the conditional flow's
                     21, giving the 24-channel unconditional steps.
      walk level 1   checkpoint block ``flow.layers.1..13`` + ``level0_condFlow``
                     z enters as 6 channels (level 0's unsqueeze2d output).  The
                     Split consumes those 6 and cats on the conditional flow's
                     6, giving the 12-channel steps, which the final
                     unsqueeze2d turns back into 3 channels at HR.

    Per level, the reverse walk is exactly

        feature = cond_feature(cat([z, upsample2x(prev_feature)]))   # 128ch
        a       = f(feature)                                      # cond_f_out
        z       = sample(mean + exp(eps_std * logs) * eps)         # cond_in
        z       = cond_steps_reverse(z, u=feature)                 # cond_in
        z       = cat([z_entry, z])                                # full
        13 x step_reverse(z)
        z       = unsqueeze2d(z)                                   # next level

    Every width below is READ from the checkpoint rather than derived: the
    checkpoint is the only authority on the geometry, and this port has been
    burned once already by a channel count that was reasoned out instead of
    measured.
    """
    if level == 0:                      # checkpoint level 1: 3 -> 24 channels
        layers = range(16, 29)
        cond = "flow.level1_condFlow"
        cond_levels = 0                 # num_levels_condition
    else:                               # checkpoint level 0: 6 -> 12 channels
        layers = range(1, 14)
        cond = "flow.level0_condFlow"
        cond_levels = 1

    cond_ch = 128                       # RRDB_nf * num_features_condition = 64 * 2
    in_ch = int(ck.raw(f"{cond}.conv_first.weight").shape[1]) - cond_levels * cond_ch
    c_split = in_ch                     # the Split consumes the whole entry z
    full = int(ck.raw(f"flow.layers.{layers[0]}.actnorm.bias").shape[1])
    cond_f_out = int(ck.raw(f"{cond}.f.weight").shape[0])
    cond_in = cond_f_out // 2
    assert full == in_ch + cond_in, (full, in_ch, cond_in)

    # The unconditional steps must be full-width, and the conditional ones
    # cond_in-wide, or the two halves of this function do not describe the same
    # graph.
    for i in layers:
        got = int(ck.raw(f"flow.layers.{i}.actnorm.bias").shape[1])
        assert got == full, (i, got, full)
    n_cond_steps = len({k.split(".additional_flow_steps.")[1].split(".")[0]
                        for k in ck.sd if k.startswith(f"{cond}.additional_flow_steps.")})
    for i in range(n_cond_steps):
        got = int(ck.raw(f"{cond}.additional_flow_steps.{i}.actnorm.bias").shape[1])
        assert got == cond_in, (i, got, cond_in)
        c1 = int(ck.raw(f"{cond}.additional_flow_steps.{i}.affine.f.conv1.weight").shape[1])
        assert c1 == cond_in // 2 + cond_ch, (i, c1)

    steps = [flow_step(ck, f"flow.layers.{idx}", False) for idx in layers]
    cond_steps = [flow_step(ck, f"{cond}.additional_flow_steps.{i}", False)
                  for i in range(n_cond_steps)]

    trunk_base = [f"{cond}.RRDB_trunk0", f"{cond}.RRDB_trunk1"]
    out = {
        "steps": steps,
        "cond_steps": cond_steps,
        "cond_conv_first_w": ck.get(f"{cond}.conv_first.weight"),
        "cond_conv_first_b": ck.get(f"{cond}.conv_first.bias").reshape(-1),
        # trunk_conv1 has no ActNorm of its own in this checkpoint, so it is
        # emitted verbatim; its output is added to the first trunk block's
        # input, as get_conditional_feature_SR does.
        "trunk_conv_w": ck.get(f"{cond}.trunk_conv1.weight"),
        "trunk_conv_b": ck.get(f"{cond}.trunk_conv1.bias").reshape(-1),
        "trunk0": rrdb_block(ck, trunk_base[0]),
        "trunk1": rrdb_block(ck, trunk_base[1]),
        # The conditional flow's output layer, Conv2dZeros, keeps its
        # logscale_factor folded in, so the engine reads mean = y[0::2] and
        # logs = y[1::2] off one convolution's output.
        "cond_out_w": None,   # filled below
        "cond_out_b": None,
    }
    ow = ck.get(f"{cond}.f.weight")
    ob = ck.get(f"{cond}.f.bias").reshape(-1)
    ol = ck.get(f"{cond}.f.logs").reshape(-1)
    ow, ob = fold_scale(ow, ob, ol * 3.0)
    out["cond_out_w"] = ow
    out["cond_out_b"] = ob
    geom = {"in_ch": in_ch, "c_split": c_split, "full_ch": full,
            "cond_in": cond_in, "cond_out": cond_f_out, "cond_ch": cond_ch,
            "cond_levels": cond_levels, "steps": len(steps),
            "cond_steps": len(cond_steps)}
    return out, geom


def expected_geometry(scale, ck):
    """Derive the walk's per-level shapes from the checkpoint itself.

    Nothing here is hard-coded: every number is read out of the tensors, so a
    mismatched architecture fails loudly at conversion time rather than at
    inference time."""
    assert scale == 4, "this converter is written for the x4 SR model"
    assert ck.has("flow.level1_condFlow.conv_first.weight")
    assert ck.has("flow.level0_condFlow.conv_first.weight")
    keys = ("steps", "cond_steps", "in_ch", "c_split", "full_ch", "cond_in",
            "cond_out", "cond_ch", "cond_levels")
    g = {k: [] for k in keys}
    g["n_levels"] = 2
    for level in (0, 1):
        _, geom = build(ck, level)
        for k in keys:
            g[k].append(geom[k])
    assert g["in_ch"] == [3, 6], g["in_ch"]
    assert g["full_ch"] == [24, 12], g["full_ch"]
    assert g["cond_in"] == [21, 6], g["cond_in"]
    assert g["cond_levels"] == [0, 1], g["cond_levels"]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--scale", type=int, default=4, choices=(4,))
    ap.add_argument("--check-only", action="store_true",
                    help="run the folds' numerical checks and write nothing")
    ap.add_argument("--eps-std", type=float, default=0.9, help="recorded only")
    args = ap.parse_args()

    print(f"reading {args.checkpoint}")
    ck = Ck(args.checkpoint)
    print(f"  {len(ck.sd)} tensors")

    verify_folds(ck)

    if args.check_only:
        return 0

    geom = expected_geometry(args.scale, ck)
    tensors = {}
    for level in (0, 1):
        params, g = build(ck, level)
        name = f"l{level}"
        for k, v in to_tensors(params).items():
            tensors[f"{name}.{k}"] = v

    if args.out is None:
        raise SystemExit("no output path given")

    meta = {
        "format": FORMAT,
        "scale": str(args.scale),
        "n_levels": str(geom["n_levels"]),
        "eps_std": str(args.eps_std),
        "source": os.path.basename(args.checkpoint),
    }
    # One comma-separated list per multi-valued field, indexed by ENGINE level
    # (walk order), not by the checkpoint's build order.
    for key in ("in_ch", "c_split", "full_ch", "cond_in", "cond_out", "cond_ch",
                "cond_levels", "steps", "cond_steps"):
        meta[key] = ",".join(str(v) for v in geom[key])
    write_safetensors(args.out, tensors, meta)
    return 0


def write_safetensors(path, tensors, meta):
    offset = 0
    payloads = []
    header = {"__metadata__": dict(meta)}
    for name in sorted(tensors):
        t = np.ascontiguousarray(tensors[name], dtype=np.float32)
        raw = t.tobytes()
        pad = (-offset) % ALIGN
        if pad:
            payloads.append(b"\x00" * pad)
            offset += pad
        header[name] = {"dtype": "F32", "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        payloads.append(raw)
        offset += len(raw)
    header["__metadata__"]["tensor_count"] = str(len(tensors))
    hjson = json.dumps(header, separators=(",", ":")).encode()
    hjson += b" " * ((-(len(hjson) + 8)) % ALIGN)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hjson)))
        fh.write(hjson)
        for p in payloads:
            fh.write(p)
    mb = os.path.getsize(path) / (1024 * 1024)
    print(f"wrote {path} ({mb:.1f} MiB, {len(tensors)} tensors)")


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def check(name, got, want, tol, report):
    d = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
    rel = d / max(1.0, float(np.max(np.abs(want.astype(np.float64)))))
    ok = d <= tol or rel <= tol
    report.append((name, d, rel, tol, ok))
    return ok


def verify_folds(ck):
    """Prove each fold on real weights and real activations.

    The point of these is that a fold which is algebraically right and
    numerically wrong is invisible until an image comes out wrong, and then it
    is indistinguishable from a hundred other causes.
    """
    rng = np.random.default_rng(1234)
    report = []
    print("\nfold checks (max |diff| against the unfolded reference)")

    # 1. conv1/conv2 of a coupling net: (conv + actnorm) vs folded conv.
    for tag, base in (("L0 step1", "flow.layers.1"), ("L1 step16", "flow.layers.16"),
                      ("L0 cond0", "flow.level0_condFlow.additional_flow_steps.0"),
                      ("L1 cond0", "flow.level1_condFlow.additional_flow_steps.0")):
        for which in ("conv1", "conv2"):
            w = ck.get(f"{base}.affine.f.{which}.weight")
            ab = ck.get(f"{base}.affine.f.{which}.actnorm.bias").reshape(-1)
            al = ck.get(f"{base}.affine.f.{which}.actnorm.logs").reshape(-1)
            x = rng.standard_normal((2, w.shape[1], 7, 5), dtype=np.float32)
            if w.shape[2] == 3:
                want = ref_actnorm(ref_conv2d(x, w, None, pad=1), ab, al)
                fw, fb = fold_actnorm_out(w, None, ab, al)
                got = ref_conv2d(x, fw, fb, pad=1)
            else:
                want = ref_actnorm(ref_conv1x1(x, w), ab, al)
                fw, fb = fold_actnorm_out(w, None, ab, al)
                got = ref_conv1x1(x, fw, fb)
            check(f"{tag} {which} actnorm folded", got, want, 1e-5, report)

    # 2. Conv2dZeros' logscale factor.
    for tag, base in (("L0 step1", "flow.layers.1"), ("L1 cond0", "flow.level1_condFlow.additional_flow_steps.0")):
        w = ck.get(f"{base}.affine.f.conv3.weight")
        b = ck.get(f"{base}.affine.f.conv3.bias").reshape(-1)
        l = ck.get(f"{base}.affine.f.conv3.logs").reshape(-1)
        x = rng.standard_normal((2, w.shape[1], 6, 4), dtype=np.float32)
        want = ref_conv2d(x, w, b, pad=1) * np.exp(l.reshape(1, -1, 1, 1) * 3.0)
        fw, fb = fold_scale(w, b, l * 3.0)
        got = ref_conv2d(x, fw, fb, pad=1)
        check(f"{tag} conv3 logscale folded", got, want, 1e-5, report)

    # 3. The 1x1 inverse.
    for tag, base in (("L0 step1", "flow.layers.1"), ("L1 step28", "flow.layers.28"),
                      ("L1 cond0", "flow.level1_condFlow.additional_flow_steps.0")):
        w = ck.get(f"{base}.permute.weight")
        wr = torch.inverse(torch.from_numpy(w).double()).float().numpy()
        eye = w.astype(np.float64) @ wr.astype(np.float64)
        check(f"{tag} 1x1 inverse (W @ W^-1)", eye, np.eye(w.shape[0]), 1e-5, report)

    # 4. The WHOLE step: forward on the reference, then reverse on the folded
    #    path, must return the input.  This is the check that catches an order
    #    mistake (coupling/permute/actnorm reversed wrongly) rather than a fold.
    #    The 21-channel conditional step at engine level 0 is included on
    #    purpose: its ODD width makes the coupling's halves 10 and 11 and its
    #    conv3 emit 2 * (21 - 10) = 22 channels, which every even-width
    #    assumption gets wrong.
    for tag, base, cz, cch in (("L0 step1", "flow.layers.1", 12, 0),
                               ("L1 step16", "flow.layers.16", 24, 0),
                               ("L1 cond0 (21ch)", "flow.level1_condFlow.additional_flow_steps.0", 21, 128),
                               ("L0 cond0 (6ch)", "flow.level0_condFlow.additional_flow_steps.0", 6, 128)):
        x = rng.standard_normal((2, cz, 7, 5), dtype=np.float32)
        # A conditional step reads a 128-channel feature alongside its first
        # half, so the round trip has to carry one through both directions.
        u = None if cch == 0 else rng.standard_normal((2, cch, 7, 5), dtype=np.float32)
        z = flow_step_fwd_reference(ck, base, x, u)
        step = flow_step(ck, base, False)
        back = flow_step_rev_folded(step, z, u)
        check(f"{tag} step forward->reverse round trip", back, x, 1e-4, report)

    # 6. The conditional output layer's logscale fold.
    w = ck.get("flow.level1_condFlow.f.weight")
    b = ck.get("flow.level1_condFlow.f.bias").reshape(-1)
    l = ck.get("flow.level1_condFlow.f.logs").reshape(-1)
    x = rng.standard_normal((2, w.shape[1], 6, 4), dtype=np.float32)
    want = ref_conv2d(x, w, b, pad=1) * np.exp(l.reshape(1, -1, 1, 1) * 3.0)
    fw, fb = fold_scale(w, b, l * 3.0)
    check("L1 cond out logscale folded", ref_conv2d(x, fw, fb, pad=1), want, 1e-5, report)

    bad = [r for r in report if not r[4]]
    for (name, d, rel, tol, ok) in report:
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<44} max |d| {d:.3e} (rel {rel:.2e}, tol {tol:.0e})")
    if bad:
        raise SystemExit(f"{len(bad)} fold check(s) FAILED")
    print(f"  all {len(report)} fold checks passed")


if __name__ == "__main__":
    sys.exit(main())
