//! The HCFlow graph, on the host: planes, the CPU reference, and the shared
//! traversal that the CUDA backend mirrors.
//!
//! WHAT HCFLOW DOES AT INFERENCE. The checkpoint is a two-level conditional
//! normalizing flow for x4 super-resolution, and inference walks its layer list
//! BACKWARDS. `z` starts as the LR image itself (3 channels) and each level, in
//! walk order (the reference's 24-channel block first, then its 12-channel one):
//!
//! 1. builds a CONDITIONAL FEATURE from `cat([z, upsample2x(feature_below)])` -
//!    `conv_first`, two RRDB trunks of dense blocks, then `trunk_conv1` and a
//!    residual, all at 64 channels - and because the RRDB pyramid ends at 128
//!    channels it is `cat([trunk0, trunk_conv1(trunk1) + conv_first_out])`;
//! 2. samples the latent: `h = f(feature)` is read as mean = h's even channels
//!    and logs = h's odd ones, and `z = mean + exp(eps_std * logs) * eps`. `eps`
//!    is the caller's randomness: with `eps_std = 0` the map is deterministic
//!    and the engine is a plain feed-forward network;
//! 3. runs that level's conditional flow steps IN REVERSE, each coupling on
//!    `cat([z1, feature])`: coupling, then the inverse 1x1, then the ActNorm's
//!    reverse;
//! 4. applies the Split inverse, `z = cat([z, a])`, which is what brings `z` up
//!    to the width the unconditional steps run at;
//! 5. runs thirteen unconditional flow steps in reverse;
//! 6. applies Unsqueeze2d: 4x channels out of half the resolution, i.e. the
//!    latent becomes the next level's input (or, at the last level, the RGB
//!    image).
//!
//! Steps 1-6 are one `ConditionalFlow` plus one layer list, and the ORDER is
//! the reference's, not a choice: the conditional flow samples the channels the
//! Split CONSUMED, and its output lands AFTER them in the concatenation.
//!
//! The per-step order matters and is the reference's, not a choice: the forward
//! direction is ActNorm -> 1x1 -> coupling, so the reverse visits them last to
//! first, and the 1x1 uses the INVERSE matrix that the converter computed once
//! in float64.
//!
//! This module owns the algebra and the CPU implementation. `gpu.rs` implements
//! the same functions against the CUDA kernels and is checked against this one.

use crate::weights::{cond_step, step, Config, Weights};
use lightgpu::ops::cpu;

/// Everything about a run that is not in the checkpoint: the sampling
/// temperature and the noise itself.
///
/// `eps` is the caller's randomness, one plane per level with `cond_in[level]`
/// channels at the level's resolution. The reference draws it from
/// `N(0, eps_std^2)`; the caller supplies it explicitly so a run is
/// reproducible (and so the self-test can compare the two backends on the SAME
/// draw rather than hoping two random draws agree).
#[derive(Default)]
pub struct Options {
    pub eps_std: f32,
    pub eps: Option<Vec<Plane>>,
}

impl Options {
    /// A run at the checkpoint's own temperature, sampling normally.
    ///
    /// AT ZERO THIS IS NOT A DRAW: `eps` stays absent, the latent equation
    /// contributes a zero plane, and the result is the mean map. That is what
    /// makes `eps_std 0` the deterministic mode rather than a very quiet sample.
    pub fn sampled(eps_std: f32) -> Options {
        Options { eps_std, eps: None }
    }

    /// A run whose noise is supplied, for reproducibility.
    pub fn with_eps(eps_std: f32, eps: Vec<Plane>) -> Options {
        Options { eps_std, eps: Some(eps) }
    }
}

/// A plane of f32 in NCHW order, with the shape it is to be read as.
#[derive(Clone, Debug)]
pub struct Plane {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(c: usize, h: usize, w: usize) -> Plane {
        Plane { c, h, w, data: vec![0.0; c * h * w] }
    }

    pub fn from_vec(c: usize, h: usize, w: usize, data: Vec<f32>) -> Plane {
        assert_eq!(data.len(), c * h * w);
        Plane { c, h, w, data }
    }

    pub fn hw(&self) -> usize {
        self.h * self.w
    }
}

/// The reference's coupling log-scale: `logscale = 0.318 * atan(2 * scale)`.
///
/// The constant and the arctan are the reference's, typed as the reference typed
/// them (HCFlow `ConditionalFlow.py` and `AffineCouplings.py`). Both the CPU twin
/// and the CUDA path must use exactly this, because a different but "equivalent"
/// clamp or scaling changes the sampled latent and therefore the image.
///
/// `0.318` is NOT an approximation of `1/pi` to be tidied up: the reference's
/// literal is 0.318, `1/pi` would differ in the third decimal, and clippy's
/// `approx_constant` suggestion is therefore a behaviour change here.
#[allow(clippy::approx_constant)]
#[inline]
pub fn logscale_of(scale: f32) -> f32 {
    0.318 * (2.0 * scale).atan()
}

// ---------------------------------------------------------------------------
// primitive ops, CPU. These mirror the CUDA kernels one for one, and the CUDA
// backend is validated against them.
// ---------------------------------------------------------------------------

/// 3x3, stride 1, pad 1, with a nullable bias. `w` is [c_out][c_in][3][3].
pub fn conv3x3(
    x: &Plane,
    w: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
) -> Plane {
    let mut y = Plane::new(c_out, x.h, x.w);
    cpu::conv3x3s1p1(&x.data, w, bias.unwrap_or(&[]), &mut y.data, x.c, c_out, x.h, x.w);
    y
}

/// 1x1, with a nullable bias. `w` is [c_out][c_in] or [c_out][c_in][1][1].
pub fn conv1x1(x: &Plane, w: &[f32], bias: Option<&[f32]>, c_out: usize) -> Plane {
    let mut y = Plane::new(c_out, x.h, x.w);
    cpu::conv1x1(&x.data, w, bias.unwrap_or(&[]), &mut y.data, x.c, c_out, x.h, x.w);
    y
}

/// `out = in * scale[c] + shift[c]`, in place, with a nullable shift.
pub fn channel_affine(x: &mut Plane, scale: &[f32], shift: Option<&[f32]>) {
    let hw = x.hw();
    for c in 0..x.c {
        let s = scale[c];
        let b = shift.map(|v| v[c]).unwrap_or(0.0);
        let p = &mut x.data[c * hw..(c + 1) * hw];
        for v in p.iter_mut() {
            *v = *v * s + b;
        }
    }
}

/// ReLU in place.
pub fn relu(x: &mut Plane) {
    for v in x.data.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Nearest-neighbour 2x upsample, the PyTorch `floor(o/2)` contract.
pub fn upsample2x(x: &Plane) -> Plane {
    let mut y = Plane::new(x.c, x.h * 2, x.w * 2);
    let (xhw, yhw, yw) = (x.hw(), y.hw(), y.w);
    let (yh, ywd) = (y.h, y.w);
    for c in 0..x.c {
        for oy in 0..yh {
            for ox in 0..ywd {
                y.data[c * yhw + oy * yw + ox] =
                    x.data[c * xhw + (oy / 2) * x.w + (ox / 2)];
            }
        }
    }
    y
}

/// The pixel-unshuffle permutation: [c][h][w] -> [4c][h/2][w/2] with output
/// channel `c*4 + dy*2 + dx`.
///
/// The graph's own squeeze is `conv2d_0`'s stride-2 conv; this exists as the CPU
/// twin of the CUDA `pixel_unshuffle2` kernel, which the primitive self-test
/// checks against it, so it is compiled only for the build that has that kernel.
#[cfg(feature = "cuda")]
pub fn pixel_unshuffle2(x: &Plane) -> Plane {
    let mut y = Plane::new(x.c * 4, x.h / 2, x.w / 2);
    let (oh, ow) = (y.h, y.w);
    let iplane = x.hw();
    let oplane = y.hw();
    for c in 0..x.c {
        for dy in 0..2 {
            for dx in 0..2 {
                let oc = c * 4 + dy * 2 + dx;
                for yy in 0..oh {
                    for xx in 0..ow {
                        y.data[oc * oplane + yy * ow + xx] =
                            x.data[c * iplane + (2 * yy + dy) * x.w + (2 * xx + dx)];
                    }
                }
            }
        }
    }
    y
}

/// The inverse of [`pixel_unshuffle2`]: [c][h][w] -> [c/4][2h][2w]. This is the
/// reference's `Unsqueeze2d` in its reverse direction, applied to the latent to
/// produce the level's output.
pub fn unsqueeze2d(x: &Plane) -> Plane {
    assert_eq!(x.c % 4, 0);
    let oc = x.c / 4;
    let mut y = Plane::new(oc, x.h * 2, x.w * 2);
    let iplane = x.hw();
    let oplane = y.hw();
    for c in 0..oc {
        for dy in 0..2usize {
            for dx in 0..2usize {
                let ic = c * 4 + dy * 2 + dx;
                for yy in 0..x.h {
                    for xx in 0..x.w {
                        y.data[c * oplane + (2 * yy + dy) * y.w + (2 * xx + dx)] =
                            x.data[ic * iplane + yy * x.w + xx];
                    }
                }
            }
        }
    }
    y
}

/// Concatenate planes of equal spatial size along the channel axis.
pub fn cat(a: &Plane, b: &Plane) -> Plane {
    assert_eq!((a.h, a.w), (b.h, b.w));
    let mut y = Plane::new(a.c + b.c, a.h, a.w);
    let hw = a.hw();
    y.data[..a.c * hw].copy_from_slice(&a.data);
    y.data[a.c * hw..].copy_from_slice(&b.data);
    y
}

/// The x-in-place residual add `y += x`.
pub fn add_into(y: &mut Plane, x: &Plane) {
    for (d, s) in y.data.iter_mut().zip(x.data.iter()) {
        *d += *s;
    }
}

/// `y = x * s`, out of place (the RDB's `out * 0.2`).
pub fn scaled(x: &Plane, s: f32) -> Plane {
    Plane { c: x.c, h: x.h, w: x.w, data: x.data.iter().map(|v| v * s).collect() }
}


// ---------------------------------------------------------------------------
// the flow's pieces
// ---------------------------------------------------------------------------

/// One flow step's weights, borrowed from the checkpoint. Every field is a
/// slice the caller already validated at load, so a step cannot be built with a
/// mismatched width without having failed earlier.
pub struct Step<'a> {
    /// The ActNorm's REVERSE pair: `an_scale = exp(-logs)` and
    /// `an_bias = -bias`, as the converter writes them. The engine only ever
    /// runs the ActNorm backwards, so the reversal is done once at convert time
    /// rather than per step here.
    pub an_scale: &'a [f32],
    pub an_bias: &'a [f32],
    /// The INVERSE 1x1, flattened [c][c].
    pub w_rev: &'a [f32],
    pub c1w: &'a [f32],
    pub c1b: &'a [f32],
    pub c2w: &'a [f32],
    pub c2b: &'a [f32],
    pub c3w: &'a [f32],
    pub c3b: &'a [f32],
    /// Channels of the coupling's input: `z1` plus the conditioning, when any.
    pub half: usize,
    /// Channels of the coupling's input z (the coupling's two halves add up to
    /// this; odd widths split as `n / 2` and `n - n / 2`).
    pub out_ch: usize,
    pub hidden: usize,
}

/// Gather one step's slices out of the named prefix.
///
/// `out_ch` is the step's `z` width and `half` its coupling's first-half width;
/// `cond` is the conditioning width the coupling's `conv1` sees appended to
/// `z1` - 0 for an unconditional step and the cached feature's 128 channels for
/// a conditional one. It is a parameter rather than something derived because
/// the two kinds of step are otherwise identical and `c1w`'s input channels are
/// the only tensor that differs.
pub fn load_step<'a>(
    w: &'a Weights,
    base: &str,
    half: usize,
    out_ch: usize,
    cond: usize,
) -> Result<Step<'a>, String> {
    let g = |suffix: &str, want: usize| -> Result<&'a [f32], String> {
        let name = format!("{base}.{suffix}");
        let v = w.get(&name)?;
        if v.len() != want {
            return Err(format!("{name}: {} elements, expected {want}", v.len()));
        }
        Ok(v)
    };
    let hidden = w.config.hidden;
    Ok(Step {
        an_scale: g("an_scale", out_ch)?,
        an_bias: g("an_bias", out_ch)?,
        w_rev: g("w_rev", out_ch * out_ch)?,
        c1w: g("c1w", hidden * (half + cond) * 9)?,
        c1b: g("c1b", hidden)?,
        c2w: g("c2w", hidden * hidden)?,
        c2b: g("c2b", hidden)?,
        // The coupling emits one shift/scale pair per channel of its SECOND
        // half, which is the longer one when the step's width is odd.
        c3w: g("c3w", 2 * (out_ch - half) * hidden * 9)?,
        c3b: g("c3b", 2 * (out_ch - half))?,
        half,
        out_ch,
        hidden,
    })
}

/// The coupling's reverse direction, written into `out` so the caller needs no
/// scratch plane.
///
/// Forward the reference computes `h = F(cat([z1, cond]))`, splits `h` with
/// `split_feature(h, "cross")` - which is `shift = h[:, 0::2]` and
/// `scale = h[:, 1::2]`, i.e. the EVEN channel is the shift and the ODD one the
/// scale - and updates `z2 = (z2 + shift) * exp(0.318 * atan(2 * scale))`. The
/// reverse inverts that update and nothing else: `z1` is copied through, and
/// `z2 = z2 * exp(-logscale) - shift`.
///
/// `z2` is the LONGER half when the step's width is odd (a 21-channel
/// conditional step splits 10 / 11), which is why `half` and the coupling's
/// output width are read from the step rather than assumed equal.
pub fn coupling_rev(step: &Step, z: &Plane, cond: Option<&Plane>, out: &mut Plane) {
    let (z1, z2) = split_half(z, step.half);
    let h_in = match cond {
        // cat([z1, cond]) along channels; the coupling's conv1 expects that
        // order, and the converter's c1w shapes were checked against it.
        Some(c) => cat_planes(&z1, c, z.h, z.w),
        None => z1.clone(),
    };
    let mut h = conv3x3(&h_in, step.c1w, Some(step.c1b), step.hidden);
    relu(&mut h);
    let mut h2 = conv1x1(&h, step.c2w, Some(step.c2b), step.hidden);
    relu(&mut h2);
    h = conv3x3(&h2, step.c3w, Some(step.c3b), step.coupling_out());

    let hw = h.hw();
    let n2 = z2.c;
    for i in 0..n2 {
        let shift = &h.data[(2 * i) * hw..(2 * i + 1) * hw];
        let scale = &h.data[(2 * i + 1) * hw..(2 * i + 2) * hw];
        let inp = &z2.data[i * hw..(i + 1) * hw];
        let dst = &mut out.data[(step.half + i) * hw..(step.half + i + 1) * hw];
        for p in 0..hw {
            dst[p] = inp[p] * (-logscale_of(scale[p])).exp() - shift[p];
        }
    }
    out.data[..step.half * hw].copy_from_slice(&z1.data);
}

/// Split a plane's channels into two halves at `split`.
pub fn split_half(x: &Plane, split: usize) -> (Plane, Plane) {
    let hw = x.hw();
    let a = Plane::from_vec(split, x.h, x.w, x.data[..split * hw].to_vec());
    let b = Plane::from_vec(x.c - split, x.h, x.w, x.data[split * hw..].to_vec());
    (a, b)
}

/// cat([a, b]) where both are planes of the same spatial size. `cat` above does
/// the same for two borrowed planes.
fn cat_planes(a: &Plane, b: &Plane, h: usize, w: usize) -> Plane {
    let mut y = Plane::new(a.c + b.c, h, w);
    let hw = y.hw();
    y.data[..a.c * hw].copy_from_slice(&a.data);
    y.data[a.c * hw..].copy_from_slice(&b.data);
    y
}

/// The full reverse of one flow step: coupling, then the inverse 1x1, then the
/// ActNorm's reverse `x * exp(-logs) - bias`.
pub fn flow_step_rev(step: &Step, z: &Plane, cond: Option<&Plane>) -> Plane {
    let mut out = Plane::new(z.c, z.h, z.w);
    coupling_rev(step, z, cond, &mut out);
    let mut y = conv1x1(&out, step.w_rev, None, step.out_ch);
    // `an_scale`/`an_bias` are already exp(-logs) / -bias (see the converter);
    // re-deriving them here would be a second place for the direction to go
    // wrong.
    channel_affine(&mut y, step.an_scale, Some(step.an_bias));
    y
}


// ---------------------------------------------------------------------------
// the conditional feature: conv_first, two RRDB trunks, trunk_conv1 + residual
// ---------------------------------------------------------------------------

/// One ResidualDenseBlock: five dense 3x3 convs with leaky-ReLU(0.2) after all
/// but the last, then `out * 0.2 + x`.
///
/// The concat of every previous output is what makes it DENSE: with a hidden
/// width nf and a growth channel gc, conv1 reads nf channels, conv2 reads
/// nf + gc, conv3 nf + 2*gc, conv4 nf + 3*gc, and conv5 nf + 4*gc, and only
/// conv5 returns to nf.
///
/// The residual at BOTH levels (`ResidualDenseBlock.forward` ends with
/// `x5 * 0.2 + x` and `RRDB.forward` with `out * 0.2 + x`) is what keeps the
/// trunk's signal near its input scale: without them the stack multiplies.
pub fn rdb(x: &Plane, w: &[&[f32]], b: &[&[f32]], nf: usize, gc: usize) -> Plane {
    let mut feats: Vec<Plane> = Vec::with_capacity(4);
    let mut cat_in = x.clone();
    let mut last = None;
    for i in 0..5 {
        let outc = if i < 4 { gc } else { nf };
        let mut h = conv3x3(&cat_in, w[i], Some(b[i]), outc);
        if i < 4 {
            leaky_relu(&mut h, 0.2);
            feats.push(h);
            cat_in = concat_all(x, &feats);
        } else {
            last = Some(h);
        }
    }
    // The dense stack ends with its OWN `x5 * 0.2 + x` - the ResidualDenseBlock
    // in the reference carries a residual, and the enclosing RRDB carries a
    // second one, so both 0.2 scalings are real and neither is optional: a
    // missing pair turns the trunk into a multiplicative amplifier (a small
    // constant input reached 1e16 by the end of the stack) rather than the
    // scale-preserving feature extractor it is.
    let mut y = scaled(&last.unwrap(), 0.2);
    add_into(&mut y, x);
    y
}

/// `cat([x, f0, f1, ...])` along channels.
fn concat_all(x: &Plane, feats: &[Plane]) -> Plane {
    let total: usize = x.c + feats.iter().map(|f| f.c).sum::<usize>();
    let mut y = Plane::new(total, x.h, x.w);
    let hw = x.hw();
    let mut off = 0;
    y.data[off..off + x.c * hw].copy_from_slice(&x.data);
    off += x.c * hw;
    for f in feats {
        y.data[off..off + f.c * hw].copy_from_slice(&f.data);
        off += f.c * hw;
    }
    y
}

#[inline]
pub fn leaky_relu(x: &mut Plane, slope: f32) {
    for v in x.data.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// One RRDB: three RDBs in series with a single outer residual,
/// `out * 0.2 + x`.
pub fn rrdb(x: &Plane, w: &[&[f32]], b: &[&[f32]], nf: usize, gc: usize) -> Plane {
    let mut h = x.clone();
    for r in 0..3 {
        h = rdb(&h, &w[r * 5..r * 5 + 5], &b[r * 5..r * 5 + 5], nf, gc);
    }
    // `RRDB.forward` is `out * 0.2 + x`: the RESIDUAL BRANCH is scaled, not the
    // input. Writing it the other way round (`0.2 * x + h`) is algebraically
    // similar and numerically nothing like it - the branch output is ~1 per
    // channel while the input is ~0.1, so the transposed version adds a
    // full-scale branch unscaled and the stack grows 25x per trunk.
    let mut y = scaled(&h, 0.2);
    add_into(&mut y, x);
    y
}

/// One trunk: `nb` RRDBs in series. The block count comes from the checkpoint
/// (`rrdb_blocks`), never from the config's `RRDB_nb`.
pub fn trunk(
    x: &Plane,
    w: &Weights,
    level: usize,
    t: usize,
    nf: usize,
    gc: usize,
) -> Result<Plane, String> {
    let mut cur = x.clone();
    for blk in 0..w.config.rrdb_blocks {
        let base = format!("l{level}.trunk{t}.blocks.{blk}");
        let mut wv: Vec<&[f32]> = Vec::with_capacity(15);
        let mut bv: Vec<&[f32]> = Vec::with_capacity(15);
        for r in 1..=3 {
            for c in 1..=5 {
                wv.push(w.get(&format!("{base}.{}.w{c}", r - 1))?);
                bv.push(w.get(&format!("{base}.{}.b{c}", r - 1))?);
            }
        }
        cur = rrdb(&cur, &wv, &bv, nf, gc);
    }
    Ok(cur)
}

// ---------------------------------------------------------------------------
// the conditional flow (ConditionalFlow in the reference)
//
// Reference `get_conditional_feature_SR`, in the direction inference uses:
//
//   f_first = conv_first(cat([z, upsample2x(feature_below)]));
//   f1      = trunk0(f_first);
//   f2      = trunk1(f1);
//   feature = cat([f1, f2])                     <- 2 * nf = 128 channels,
//   y       = f(feature);  mean = y[0::2], logs = y[1::2];
//   z       = mean + exp(eps_std * logs) * eps;   <- the sampled latent
//   z       = 13 reverse steps, each coupling on cat([z1, feature]).
//
// The residual that the reference applies is `trunk_conv1(trunk1) + f_first`,
// and `f1` is the FIRST trunk's output, so the concatenation is
// `cat([trunk0, trunk_conv1(trunk1) + f_first])`.
// ---------------------------------------------------------------------------

/// The two trunk outputs, before they are concatenated.
pub fn cond_feature_sr(w: &Weights, level: usize, u: &Plane) -> Result<(Plane, Plane), String> {
    let nf = w.config.hidden;
    let gc = w.config.gc;
    let b = format!("l{level}");
    let cfw = w.get(&format!("{b}.cond_conv_first_w"))?;
    let cfb = w.get(&format!("{b}.cond_conv_first_b"))?;
    let f_first = conv3x3(u, cfw, Some(cfb), nf);
    let f1 = trunk(&f_first, w, level, 0, nf, gc)?;
    let t1 = trunk(&f1, w, level, 1, nf, gc)?;
    let tw = w.get(&format!("{b}.trunk_conv_w"))?;
    let tb = w.get(&format!("{b}.trunk_conv_b"))?;
    let mut f2 = conv3x3(&t1, tw, Some(tb), nf);
    add_into(&mut f2, &f_first);
    Ok((f1, f2))
}

/// The concatenated feature the conditional steps and `f` consume.
pub fn cond_feature128(w: &Weights, level: usize, u: &Plane) -> Result<Plane, String> {
    let (f1, f2) = cond_feature_sr(w, level, u)?;
    Ok(cat(&f1, &f2))
}

// ---------------------------------------------------------------------------
// the walk
// ---------------------------------------------------------------------------

impl<'a> Step<'a> {
    /// The coupling's geometry, as the checkpoint defines it rather than as an
    /// even-width assumption would: `half` is `n / 2` (floor) and the coupling
    /// emits `2 * (n - n / 2)`, so an odd `n` gives a 10/11 split and 22
    /// output channels. Both the CPU and the CUDA path read these fields.
    pub fn coupling_out(&self) -> usize {
        2 * (self.out_ch - self.half)
    }
}

/// Load a step whose total width is `n` (odd allowed), taking `half = n / 2`.
///
/// `n` is the width of the step's `z`, which is what every per-channel tensor
/// (the ActNorm pair, the 1x1) is sized by - and for an odd `n` the coupling
/// emits one shift/scale pair per channel of the LONGER half, i.e. `c3` is
/// `2 * (n - n / 2)` wide while `z` stays `n`. Sizing the step by `2 * half`
/// instead would read the ActNorm one channel short, which is exactly the
/// failure the 21-channel level exists to catch.
pub fn load_step_n<'a>(w: &'a Weights, base: &str, n: usize) -> Result<Step<'a>, String> {
    load_step(w, base, Config::half(n), n, 0)
}

/// The same, for a step conditioned on an `nch`-channel feature.
pub fn load_cond_step<'a>(
    w: &'a Weights,
    base: &str,
    n: usize,
    nch: usize,
) -> Result<Step<'a>, String> {
    load_step(w, base, Config::half(n), n, nch)
}

/// One conditional flow: sample the latent, then run its steps in reverse.
///
/// `z_in` is the part of `z` the Split CONSUMED in the forward direction, i.e.
/// the first `c_split` channels; `u` is the level's conditioning input. The
/// returned plane is `a`, which the Split inverse cats on AFTER `z_in`.
pub fn cond_flow(
    w: &Weights,
    level: usize,
    z_in: &Plane,
    u: &Plane,
    opts: &Options,
    eps: Option<&Plane>,
) -> Result<Plane, String> {
    Ok(cond_flow_feature(w, level, z_in, u, opts, eps)?.0)
}

/// The same, also returning the level's 128-channel conditional feature.
///
/// The reference's `reverse_flow` keeps that feature (`conditional_feature2`)
/// and hands it to the NEXT level, upsampled and concatenated with that level's
/// `z` - so it is part of the graph, not a diagnostic. Dropping it conditions
/// the inner level on the wrong number of channels, which is a graph change no
/// shape check in the loader would notice.
pub fn cond_flow_feature(
    w: &Weights,
    level: usize,
    z_in: &Plane,
    u: &Plane,
    opts: &Options,
    eps: Option<&Plane>,
) -> Result<(Plane, Plane), String> {
    let cfg = &w.config;
    let ci = cfg.cond_in[level];
    let nsteps = cfg.cond_steps[level];
    let b = format!("l{level}");

    let feature = cond_feature128(w, level, u)?;
    if feature.c != 2 * cfg.hidden {
        return Err(format!("level {level}: feature is {} channels", feature.c));
    }

    // h = f(feature); mean = even channels, logs = odd channels. The
    // logscale_factor is already folded into cond_out_w/b by the converter.
    let ow = w.get(&format!("{b}.cond_out_w"))?;
    let ob = w.get(&format!("{b}.cond_out_b"))?;
    let h = conv3x3(&feature, ow, Some(ob), 2 * ci);

    let hw = z_in.hw();
    let zero;
    let eps = match eps {
        Some(e) => {
            if e.c != ci || e.h != z_in.h || e.w != z_in.w {
                return Err(format!(
                    "level {level}: eps is {}x{}x{}, expected {ci}x{}x{}",
                    e.c, e.h, e.w, z_in.h, z_in.w
                ));
            }
            e
        }
        None => {
            zero = Plane::new(ci, z_in.h, z_in.w);
            &zero
        }
    };
    let mut z = Plane::new(ci, z_in.h, z_in.w);
    // mean = h[0::2] = channels 0, 2, 4, ...; logs = h[1::2].
    for i in 0..ci {
        let mean = &h.data[(2 * i) * hw..(2 * i + 1) * hw];
        let logs = &h.data[(2 * i + 1) * hw..(2 * i + 2) * hw];
        let e = &eps.data[i * hw..(i + 1) * hw];
        let dst = &mut z.data[i * hw..(i + 1) * hw];
        for p in 0..hw {
            dst[p] = mean[p] + (opts.eps_std * logs[p]).exp() * e[p];
        }
    }

    for i in (0..nsteps).rev() {
        let s = load_cond_step(w, &cond_step(&b, i), ci, 2 * cfg.hidden)?;
        z = flow_step_rev(&s, &z, Some(&feature));
    }
    Ok((z, feature))
}

/// The reverse of a level's steps, from the last to the first.
pub fn steps_rev(w: &Weights, level: usize, n: usize, count: usize, z: Plane, cond: Option<&Plane>)
    -> Result<Plane, String>
{
    let b = format!("l{level}");
    let mut z = z;
    for i in (0..count).rev() {
        let s = load_step_n(w, &step(&b, i), n)?;
        z = flow_step_rev(&s, &z, cond);
    }
    Ok(z)
}

/// The whole network, walked the way the reference's `reverse_flow` walks it.
///
/// `z` starts as the LR image itself (3 channels at LR). For each level, in the
/// engine's walk order (the 24-channel level first, which is the reference's
/// LAST layer in the forward sense):
///
///   1. build the level's conditional feature from `cat([z, upsample2x(feature
///      below)])` (or from `z` alone at the outermost level, which has
///      `cond_levels = 0`);
///   2. sample the conditional latent `a` and run the level's conditional steps
///      in reverse, keeping the 128-channel feature this level produced;
///   3. `z = cat([z, a])` - the Split inverse, which is what puts `z` at the
///      width the unconditional steps run at;
///   4. run the level's unconditional steps in reverse;
///   5. `z = unsqueeze2d(z)`, a 4x channel contraction into 2x the resolution;
///   6. carry the FEATURE into the next level, which conditions on it.
///
/// After the last level `z` is 3 channels at HR, and the result is clamped to
/// [0, 1].
pub fn forward_cpu(
    w: &Weights,
    lr: &Plane,
    opts: &Options,
    mut progress: impl FnMut(&str),
) -> Result<Plane, String> {
    let cfg = &w.config;
    if lr.c != 3 {
        return Err(format!("the x4 model takes 3-channel input, got {}", lr.c));
    }
    if lr.h % 4 != 0 || lr.w % 4 != 0 {
        return Err(format!("input {}x{} must be a multiple of 4 in each axis", lr.h, lr.w));
    }
    let mut z = lr.clone();
    let mut feature_prev: Option<Plane> = None;
    for level in 0..cfg.n_levels {
        let ic = cfg.in_ch[level];
        let ci = cfg.cond_in[level];
        let full = cfg.full_ch[level];
        if z.c != ic {
            return Err(format!(
                "level {level}: z has {} channels entering, the checkpoint expects {ic}",
                z.c
            ));
        }
        progress(&format!(
            "level {}/{} ({} -> {} channels)",
            level + 1,
            cfg.n_levels,
            ic,
            full
        ));

        // 1. The conditioning input: this level's own z, plus the upsampled
        //    FEATURE from the level below when the checkpoint says this level
        //    consumes one (cond_levels).
        let u = match (&feature_prev, cfg.cond_levels[level]) {
            (Some(f), n) if n > 0 => cat(&z, &upsample2x(f)),
            _ => z.clone(),
        };

        // 2/3. The conditional flow samples `a` from the z the Split consumed in
        //      the forward direction; the Split inverse cats it on afterwards.
        let eps = opts.eps.as_ref().and_then(|v| v.get(level));
        let (a, feature) = cond_flow_feature(w, level, &z, &u, opts, eps)?;
        if a.c != ci {
            return Err(format!("level {level}: conditional flow gave {} channels, expected {ci}", a.c));
        }
        z = cat(&z, &a);

        // 4. The unconditional steps run at the post-concat width.
        z = steps_rev(w, level, full, cfg.steps[level], z, None)?;

        // 5. Unsqueeze2d: 4x channels -> 2x resolution.
        z = unsqueeze2d(&z);

        // 6. The NEXT level conditions on this level's 128-CHANNEL FEATURE (the
        //    reference's `conditional_feature2`), not on the conditioning input
        //    the feature was built from: `FlowNet.reverse_flow` passes
        //    `conditional_feature2` into `cat([z, interpolate(...)])`. Carrying
        //    `u` instead silently conditions the inner level on 3 channels
        //    instead of 128.
        feature_prev = Some(feature);
    }
    progress("unsqueeze");
    let mut y = z;
    for v in y.data.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    if y.c != 3 {
        return Err(format!("output has {} channels, expected 3", y.c));
    }
    Ok(y)
}
