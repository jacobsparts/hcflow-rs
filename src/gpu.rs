//! The CUDA backend: the same graph as `net.rs`, one launch per primitive.
//!
//! This module is deliberately a transcription of `net.rs` rather than a
//! redesign. Every function there has a counterpart here with the same name (or
//! the same name plus a device handle), the same argument order and the same
//! algebra, so the two can be compared line by line and the CPU one can serve as
//! the reference the self-test checks against.
//!
//! THE BUFFER RULE. Device buffers here own the memory and hold the shape they
//! are read as, exactly like `net.Plane`; a kernel launch is the only place a
//! raw pointer appears, and it is always `buf.ptr`. Nothing is freed by hand:
//! `DevBuf` releases on drop, so a level's intermediates vanish when the level
//! returns. The buffers are allocated per level rather than for the whole pass,
//! which is what keeps the 1024x1024 case inside the recorded 929 MiB.
//!
//! WHERE THE TIME GOES. The conditional feature is two RRDB trunks of seven
//! blocks of three dense blocks of five 3x3 convolutions, i.e. 210 3x3 convs per
//! level at 64 channels, against 52 coupling convs. The whole runtime is
//! therefore the 3x3 kernel plus the concatenations that feed it, and those
//! concatenations are done with device-to-device copies (`lg_copy`) rather than
//! the host round trip the first version used - at 512x512 a host round trip is
//! 2 MiB of PCIe traffic and a pipeline drain, and there are 15 of them per
//! dense block.

use crate::cuda::{self, channel_affine, Cuda, DevBuf};
use crate::net;
use crate::net::Options;
use crate::weights::Config;
use crate::weights::{cond_step, step, rdb as rdb_name, Weights};
use lightgpu::ffi::CUdeviceptr;
use lightgpu::vm::Launch;

/// A device plane: NCHW f32 with the shape it is to be read as.
pub struct Plane {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub buf: DevBuf,
}

impl Plane {
    /// A fresh plane. THE MEMORY IS NOT ZEROED - see `alloc_zeroed`.
    pub fn alloc(c: usize, h: usize, w: usize) -> Result<Plane, String> {
        if std::env::var("HCFLOW_ZERO").is_ok() {
            return Plane::alloc_zeroed(c, h, w);
        }
        Ok(Plane { c, h, w, buf: DevBuf::alloc(c * h * w)? })
    }

    /// A plane whose bytes are all zero, for a case where the caller cannot
    /// prove it writes every element.
    ///
    /// The pool recycles blocks without clearing them and the driver makes no
    /// promise about fresh pages either, so "every caller writes what it reads"
    /// is an invariant to be tested, not assumed. `HCFLOW_ZERO=1` makes
    /// `alloc` take this path for the WHOLE graph: if a run then becomes
    /// reproducible where it was not, some kernel is reading a byte it never
    /// wrote, and the run-to-run differences are that memory's contents rather
    /// than anything numerical.
    pub fn alloc_zeroed(c: usize, h: usize, w: usize) -> Result<Plane, String> {
        Ok(Plane { c, h, w, buf: DevBuf::zeros(c * h * w)? })
    }

    pub fn from_host(v: &[f32], c: usize, h: usize, w: usize) -> Result<Plane, String> {
        if v.len() != c * h * w {
            return Err(format!("host plane is {} values, expected {}", v.len(), c * h * w));
        }
        Ok(Plane { c, h, w, buf: DevBuf::from_host(v)? })
    }

    pub fn zeros(c: usize, h: usize, w: usize) -> Result<Plane, String> {
        Ok(Plane { c, h, w, buf: DevBuf::zeros(c * h * w)? })
    }

    pub fn hw(&self) -> usize {
        self.h * self.w
    }

    pub fn numel(&self) -> usize {
        self.c * self.hw()
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.buf.ptr
    }

    pub fn to_host(&self) -> Result<Vec<f32>, String> {
        let mut v = vec![0.0f32; self.numel()];
        self.buf.download(&mut v)?;
        Ok(v)
    }
}

/// Launch geometry for the elementwise-per-pixel-per-channel kernels, which
/// share the `(x, y, z)` shape: `ceil(w / 32)`, `ceil(h / 8)`, channels.
fn pix_grid(c: usize, h: usize, w: usize) -> Launch {
    let gw = ((w as u64) + 31) / 32;
    let gh = ((h as u64) + 7) / 8;
    Launch::new((gw.max(1) as u32, gh.max(1) as u32, c.max(1) as u32), (32, 8, 1))
}

/// Launch geometry for the untiled convs.
///
/// Both kernels flatten the whole tensor themselves - they read `blockIdx.x`
/// only and compute `total = c_out * h * wd` from their arguments - so the grid
/// must be ONE dimensional over that total. Splitting the channels across
/// `gridDim.y` looks like it covers the same index space and does not: every
/// y-block would repeat the same flat range and only channel 0 would ever be
/// written, leaving the rest of the output holding uninitialised device memory.
fn conv_grid(c_out: usize, h: usize, w: usize) -> Launch {
    let total = c_out * h * w;
    Launch::new((((total + 255) / 256).max(1) as u32, 1, 1), (256, 1, 1))
}

/// Which 3x3 kernel `conv3x3` launches.
///
/// `tile` is the toolkit's shared-memory tiled kernel: a block stages a 128x8
/// output tile's input halo and its 8x4x9 weight block, and each thread holds 8
/// output-channel accumulators over 4 pixels, so one staged input value feeds 8
/// multiply-adds instead of one. The toolkit's own measurement puts it at 1845
/// GFLOP/s at 64->64 against the untiled kernel's 114, and this network is a
/// dense RRDB trunk of 210 3x3 convs per level.
///
/// `q2` is this project's own re-tiling of the same algorithm and the kernel the
/// graph runs: a TRANSPOSED weight panel so the inner loop reads its
/// output-channel weights as `float4`s instead of scalars, and a 64-column tile
/// over a 16-wide block - which both fits the level-0 planes exactly (the
/// toolkit's 128-column tile computes 128 columns on a 64-wide input and throws
/// half away) and reuses each staged input value across more threads. Measured
/// at 3075-3175 GFLOP/s on the trunk shapes against `x16`'s 2480-2779.
///
/// `x16`, `t2s`, `t2w`, `t2h`, `c8s`, `o16w`, `o16f`, `tile4`, `dbuf`, `dbuf2`
/// and the rest are the shapes that were measured while choosing, kept in
/// `this_projects_tile` because `--conv-bench` is how the next person re-measures
/// instead of guessing. They share the body, so they agree bit for bit with each
/// other (same accumulation order, same zeroed halo).
///
/// `direct` is the naive one-thread-per-output kernel, kept because the two do
/// NOT agree bit for bit: `lg_conv3x3_tile` accumulates in (ci, ky, kx) order
/// and zero-fills the halo, `lg_conv3x3s1p1` in (ky, kx, ci) order and skips
/// out-of-range taps. The difference is a few 1e-6 per conv, so every validation
/// path prints which variant it ran rather than letting the choice change in the
/// middle of a comparison.
///
/// The graph does NOT take its variant from the environment: `conv_variant`
/// returns `q2`, and `--conv-bench` sets the hook below for the duration of one
/// bench entry. The bench measures MANY variants in one process, and on sm_61
/// the driver JIT-compiles on first use, so a variant re-read from the
/// environment per launch could be re-JITted mid-timing.
static BENCH_VARIANT: std::sync::Mutex<Option<&'static str>> = std::sync::Mutex::new(None);

/// The variant `--conv-bench` is timing. Set from the bench loop only.
pub fn conv_bench_variant() -> Option<&'static str> {
    *BENCH_VARIANT.lock().unwrap()
}

/// Point the dispatch at `v` for the duration of one bench entry.
pub fn set_conv_bench_variant(v: &str) -> &'static str {
    let leak: &'static str = Box::leak(v.to_string().into_boxed_str());
    *BENCH_VARIANT.lock().unwrap() = Some(leak);
    leak
}

/// The launch geometry `--conv-bench` is timing, for the same reason.
static BENCH_GEOM: std::sync::Mutex<Option<(&'static str, usize, usize, usize, usize, usize, bool)>> =
    std::sync::Mutex::new(None);

fn set_conv_bench_geom(g: Option<(&'static str, usize, usize, usize, usize, usize, bool)>) {
    *BENCH_GEOM.lock().unwrap() = g;
}

fn conv_bench_geom() -> Option<(&'static str, usize, usize, usize, usize, usize, bool)> {
    *BENCH_GEOM.lock().unwrap()
}

pub fn conv_variant() -> &'static str {
    if let Some(v) = conv_bench_variant() {
        return v;
    }
    // `q2` is the configuration the graph dispatches to, and it is fixed here
    // rather than reachable by a name: the other variants in
    // `this_projects_tile` are the experiment sweep that `--conv-bench` runs,
    // and letting a name silently stand in for the graph's own kernel is how a
    // measurement ends up attributed to the wrong configuration. `--conv-bench`
    // sets the bench hook above; nothing else selects a variant.
    "q2"
}

/// Does this kernel want BOTH indices on blockIdx.x?
///
/// `ZSWAP = 1` in `lg_tile_body` reads the tile column from the high bits of
/// `blockIdx.x` and the output-channel group from the low ones, so the launch is
/// `(gc * gw, gh, 1)`. It exists for the graph's small dense-block shapes, where
/// the tile count is too small to fill the machine and the channel groups have to
/// make up the difference. Kept next to `this_projects_tile` rather than in it so
/// that the seven-field geometry tuple the bench harness passes around stays a
/// single shape.
fn tile_zswap(k: &str) -> bool {
    matches!(k, "lg_conv3x3_q2s" | "lg_conv3x3_catq0s")
}

/// How many output-channel groups a variant puts in ONE block on threadIdx.z.
///
/// Derived from the kernel NAME rather than carried in the geometry tuple, which
/// is deliberate: `this_projects_tile` returns a 7-tuple that four call sites
/// destructure, and a tuple field is a change to all of them plus every entry.
/// Deriving it here means a variant that is added to the table without an NG is
/// NG = 1 by construction, so the tuple and the kernel can never disagree about
/// it, and the launch is the only place that has to know.
///
/// NG = 2 is what the graph runs (both the concatenated conv and the ordinary one
/// above 160 blocks). NG = 4 was tried and is slower at every shape measured
/// (6.03 -> 5.21 -> 5.76 ms for the concatenated conv at 64->64@512x512 over
/// NG = 1/2/4), so it was removed rather than left as a bench-only variant with a
/// second set of entry points and a second set of table rows to keep correct.
fn tile_ng(k: &str) -> u32 {
    match k {
        "lg_conv3x3_q2ng2" | "lg_conv3x3_catq0ng2" => 2,
        _ => 1,
    }
}

/// This project's own tiled 3x3s: `(kernel, output columns per tile, output rows
/// per tile, block x, block y)`.
///
/// `tile4` is `lg_conv3x3_tile` with the channel blocking restructured (CI = 4,
/// transposed weight panel). The `t2*` entries are the SAME body at a narrower
/// tile: `tile4`'s 128-column tile is half wasted on a 64-wide input, and the
/// narrower tile also fits more blocks per SM. `--conv-bench` measures all of
/// them on the graph's shapes rather than one being argued for.
///
/// A row is `(kernel, output columns per tile, output rows per tile, block x,
/// block y, output channels per tile, channels on blockIdx.x)`. The last field is
/// the LAUNCH ORDER: with it false the grid is `(width, height, channels)`, which
/// puts the blocks that share an input tile a whole plane apart; with it true the
/// grid is `(channels, height, width)` so those blocks are consecutive, which is
/// what the `c16` variant exists to measure. The dispatch and `--conv-bench` both
/// read this table, so a variant cannot be benched with one geometry and run with
/// another.
pub fn this_projects_tile(variant: &str)
    -> Option<(&'static str, usize, usize, usize, usize, usize, bool)>
{
    match variant {
        "tile4" => Some(("lg_conv3x3_tile4", 128, 8, 32, 8, 8, false)),
        "t2w" => Some(("lg_conv3x3_t2w", 64, 8, 32, 8, 8, false)),
        "t2s" => Some(("lg_conv3x3_t2s", 64, 4, 32, 4, 8, false)),
        "t2h" => Some(("lg_conv3x3_t2h", 64, 16, 32, 16, 8, false)),
        "o16w" => Some(("lg_conv3x3_o16w", 64, 8, 32, 8, 16, false)),
        "o16f" => Some(("lg_conv3x3_o16f", 128, 8, 32, 8, 16, false)),
        "c8s" => Some(("lg_conv3x3_c8s", 64, 4, 32, 4, 8, false)),
        "x16" => Some(("lg_conv3x3_x16", 64, 8, 16, 8, 8, false)),
        "x8" => Some(("lg_conv3x3_x8", 64, 8, 16, 8, 8, false)),
        "w16" => Some(("lg_conv3x3_w16", 64, 16, 16, 16, 8, false)),
        "v32" => Some(("lg_conv3x3_v32", 128, 4, 32, 4, 8, false)),
        "p8" => Some(("lg_conv3x3_p8", 128, 8, 16, 8, 4, false)),
        "p8n" => Some(("lg_conv3x3_p8n", 64, 4, 16, 4, 8, false)),
        // The staging-without-arithmetic probe, at `x16`'s grid so the two can be
        // timed back to back in one process. Its GFLOP/s figure is meaningless
        // (it does 1/9 of the multiplies) - what matters is its ms, subtracted
        // from x16's.
        "stage" => Some(("lg_conv3x3_stageonly", 64, 8, 16, 8, 8, false)),
        "nold" => Some(("lg_conv3x3_stage_nold", 64, 8, 16, 8, 8, false)),
        "q1" => Some(("lg_conv3x3_q1", 64, 8, 16, 8, 8, false)),
        "q2" => Some(("lg_conv3x3_q2", 64, 8, 16, 8, 8, false)),
        // NG = 2 output-channel groups per block on threadIdx.z: the SAME tile and
        // the same per-thread arithmetic as `q2`, with the staged input tile shared
        // by two groups instead of one. `chan_x` must be TRUE: the kernel's template
        // is ZFIRST = 0, so it reads the tile column from blockIdx.z and the channel
        // group from blockIdx.x. Declaring false launches `(gw, gh, gc)` instead of
        // `(gc, gh, gw)`, and then only `min(gw, gc)` tile columns and channel groups
        // are covered - the rest of the output plane is never written. That is a
        // SILENT failure (a wrong grid permutes work, it does not fault), and it is
        // what `--ng-test`'s zeroed-output pass exists to catch.
        "q2ng2" => Some(("lg_conv3x3_q2ng2", 64, 8, 16, 8, 8, true)),
        // The concatenated-input form of the same idea, so the two can be timed
        // back to back in one process; it is what the graph's dense blocks run.
        "catq0ng2" => Some(("lg_conv3x3_catq0ng2", 64, 8, 16, 8, 8, true)),
        "q2t" => Some(("lg_conv3x3_q2t", 64, 8, 16, 8, 8, false)),
        "pipe" => Some(("lg_conv3x3_pipe", 64, 8, 16, 8, 8, false)),
        "q2x" => Some(("lg_conv3x3_q2x", 64, 8, 16, 8, 8, true)),
        "r16x" => Some(("lg_conv3x3_r16x", 32, 8, 16, 8, 16, true)),
        "r16y" => Some(("lg_conv3x3_r16y", 64, 8, 16, 8, 16, true)),
        "r32a" => Some(("lg_conv3x3_r32a", 32, 4, 32, 4, 32, true)),
        "r32b" => Some(("lg_conv3x3_r32b", 16, 8, 16, 8, 32, true)),
        "r32c" => Some(("lg_conv3x3_r32c", 32, 4, 32, 4, 32, true)),
        "dbuf" => Some(("lg_conv3x3_dbuf", 64, 8, 16, 8, 8, true)),
        "dbuf2" => Some(("lg_conv3x3_dbuf2", 64, 8, 16, 8, 8, true)),
        "p1" => Some(("lg_conv3x3_p1", 64, 8, 16, 8, 8, true)),
        "q2s" => Some(("lg_conv3x3_q2s", 64, 8, 16, 8, 8, true)),
        "pz" => Some(("lg_conv3x3_pz", 64, 8, 16, 8, 8, false)),
        "q3" => Some(("lg_conv3x3_q3", 64, 8, 16, 8, 4, false)),
        "c16" => Some(("lg_conv3x3_c16", 64, 8, 16, 8, 8, true)),
        "r16" => Some(("lg_conv3x3_r16", 64, 8, 16, 8, 8, false)),
        "r32" => Some(("lg_conv3x3_r32", 128, 8, 32, 8, 8, false)),
        "r8x" => Some(("lg_conv3x3_r8x", 64, 8, 8, 8, 8, false)),
        _ => None,
    }
}

/// Diagnostic: print the pool's live-block count with a label, when
/// `HCFLOW_LIVE_DEBUG` is set. Used to find which function is holding buffers.
pub fn dbg_live(where_: &str) {
    if std::env::var("HCFLOW_LIVE_DEBUG").is_ok() {
        let (l, f) = cuda::pool_counts();
        let (_, _, live, _, _) = cuda::pool_stats();
        eprintln!("LIVE {l:>4} blocks / {f} free, {} MiB   {where_}", live / (1024 * 1024));
    }
}


// ---------------------------------------------------------------------------
// the primitives, one per `net.rs` function
// ---------------------------------------------------------------------------

/// 3x3, stride 1, pad 1. `w` is [c_out][c_in][3][3], `bias` nullable.
pub fn conv3x3(
    ca: &Cuda,
    x: &Plane,
    w: CUdeviceptr,
    bias: Option<CUdeviceptr>,
    c_out: usize,
) -> Result<Plane, String> {
    conv3x3_act(ca, x, w, bias, c_out, 0, 0.0)
}

/// Print each DISTINCT 3x3 shape once, with its direct-form FLOP count, when
/// `HCFLOW_SHAPES` is set.
///
/// `LA_PROFILE` reports a kernel's total time per grid shape but not the CHANNEL
/// counts, and a grid of `g4x16x2` is 4x16x2 = 128 blocks either way whether the
/// conv is 64->64 or 128->32. Since the direct-form FLOP count is
/// 2 * c_in * c_out * 9 * h * w, a time without the channel counts cannot be
/// turned into a rate, and a rate is the only way to see whether a shape is
/// running at the memory wall or below it.
fn trace_shape(k: &'static str, c_in: usize, c_out: usize, h: usize, wd: usize) {
    if std::env::var("HCFLOW_SHAPES").is_err() {
        return;
    }
    thread_local! {
        static SEEN: std::cell::RefCell<std::collections::HashSet<(&'static str, usize, usize, usize, usize)>>
            = std::cell::RefCell::new(std::collections::HashSet::new());
    }
    SEEN.with(|s| {
        if s.borrow_mut().insert((k, c_in, c_out, h, wd)) {
            let gf = 2.0 * c_in as f64 * c_out as f64 * 9.0 * h as f64 * wd as f64 / 1e9;
            // The bytes the DECLARED kernel reads, which is the number that
            // matters: every (tile, channel-group) block reads the whole of its
            // input - c_in channels of a 66x10 haloed tile - no matter how few
            // output channels it computes, so the re-read factor is the
            // channel-group count. At 64->64@512x512 this gives 4096 blocks x
            // 64 x 660 x 4 = 692 MB for a 67 MB input, the 10.3x amplification
            // the kernel was profiled at.
            let staged = ((wd as f64 / 64.0).ceil()) * ((h as f64 / 8.0).ceil())
                * ((c_out as f64 / 8.0).ceil()) * c_in as f64 * 10.0 * 66.0 * 4.0
                / 1e6;
            eprintln!(
                "shape {k} {c_in}->{c_out} {h}x{wd}  {gf:.4} GFLOP  staged {staged:.1} MB"
            );
        }
    });
}

/// The same conv with an activation fused into its epilogue.
///
/// `act` is the toolkit's code - 0 none, 1 relu, 2 leaky relu with `act_p` - and
/// the tiled kernels implement it in their epilogue. The dense blocks run
/// lrelu(0.2) after each of their first four convs, so fusing it saves a launch
/// and a full pass over the plane per conv; on the `direct` variant the
/// activation is applied as the separate `lg_lrelu` launch it would otherwise
/// take, so both variants stay reachable and comparable.
pub fn conv3x3_act(
    ca: &Cuda,
    x: &Plane,
    w: CUdeviceptr,
    bias: Option<CUdeviceptr>,
    c_out: usize,
    act: i32,
    act_p: f32,
) -> Result<Plane, String> {
    cuda::tag("conv:y");
    trace_shape(&conv_variant(), x.c, c_out, x.h, x.w);
    let y = Plane::alloc(c_out, x.h, x.w)?;
    // In `--conv-bench` the geometry is whatever the bench entry set, so that
    // the kernel and the grid it is launched with always come from the same
    // place: a variant that changes the tile shape MUST NOT be timed with the
    // default grid, and reading the table here would do exactly that once the
    // table entry and the kernel list drift apart.
    let mut geom = conv_bench_geom().or_else(|| this_projects_tile(conv_variant()));
    if conv_bench_geom().is_none() {
        if let Some((_, tw, th, _, _, oc, _)) = geom {
            let gw = ((x.w + tw - 1) / tw).max(1);
            let gh = ((x.h + th - 1) / th).max(1);
            let gc = ((c_out + oc - 1) / oc).max(1);
            // NG = 2 FOR THE PLAIN CONV, where it is worth even more than on the
            // concatenated one: 6.57 -> 4.14 ms at 128->32@512x512 and 19.9 ->
            // 12.6 at 192->64 in one process (1.58x), against 1.17-1.21x for the
            // concatenated form. The concatenated conv pays for the weight
            // staging twice - its staging loop carries the plane-selection
            // chain's predicates and its float4 weight slice lands on a
            // 16-byte boundary only by luck once a group is offset by `z * OC` -
            // which is where the difference comes from. Only above 160 blocks:
            // below that there are not enough blocks to fill the SMs and NG
            // divides the count.
            if gw * gh * gc >= 160 && gc % 2 == 0 {
                geom = this_projects_tile("q2ng2");
                if std::env::var("HCFLOW_SHAPES").is_ok() {
                    eprintln!(
                        "plain-ng: {}x{} c_in {} c_out {} tiles {} gc {} blocks {} -> q2ng2",
                        x.h, x.w, x.c, c_out, gw * gh, gc, gw * gh * gc
                    );
                }
            }
            // A SECOND DECISION: is this launch too short of SPATIAL tiles to
            // fill the machine? At 64x64 the graph's dense blocks launch only
            // 32-64 blocks in total on a 20-SM part, which is where the whole of
            // LR64's deficit lives, and the narrow 32-column tile doubles the
            // tile count. MEASURED on the ordinary conv's own small shapes, one
            // process: 64->64@64x64 t2s 1644-1739 GFLOP/s against q2's
            // 1099-1344, 64->32 1152 against 819, 32->64 1781 against 1477,
            // 128->12 481 against 275, 128->42 1494 against 1117 - and at
            // 128x128 the same comparison reverses (64->64 1822 against 2514),
            // which is why the gate is the tile COUNT: at 128x128 gw*gh is 32
            // and the branch does not fire.
            // So the gate is the tile COUNT, not the plane size.
            if gw * gh < 16 {
                geom = this_projects_tile("t2s");
                if std::env::var("HCFLOW_SHAPES").is_ok() {
                    eprintln!(
                        "small-tile: {}x{} c_out {} tiles {} -> t2s",
                        x.h, x.w, c_out, gw * gh
                    );
                }
            }
        }
    }
    if let Some((k, tw, th, tbx, tby, oc, chan_x)) = geom {
        // This project's variants share the toolkit kernel's algorithm and
        // differ only in their tile shape, so the grid is derived from that
        // shape and the bounds check inside handles a partial tile.
        let gw = ((x.w + tw - 1) / tw).max(1) as u32;
        let gh = ((x.h + th - 1) / th).max(1) as u32;
        // NG GROUPS PER BLOCK. The grid's channel dimension is the number of
        // BLOCKS along it, which is the channel-group count DIVIDED by NG, and
        // the block gains the NG axis. `c_out` and `oc` are not otherwise
        // touched: the kernel computes `oc0` from the grid and threadIdx.z, and
        // its bounds predicates already handle a partial group, so a `c_out` that
        // NG does not divide exactly is still CORRECT - it just wastes the tail
        // threads of the last group. The dispatch only enables NG where it
        // divides (see the guard at the call site), so that never happens in the
        // graph, but the geometry is not relying on it.
        let ng = tile_ng(k);
        let gc = (((c_out + oc - 1) / oc).max(1) as u32 + ng - 1) / ng;
        // REACHABILITY FOR THE LAUNCH ITSELF. `HCFLOW_SHAPES=1` prints the
        // SUBSTITUTION (which variant and why) but not the GRID that variant is
        // then launched with, and a variant whose template reads blockIdx
        // differently from the table's `chan_x` writes only part of the output
        // plane - a defect a shape print alone cannot expose.
        if std::env::var("HCFLOW_LAUNCH").is_ok() {
            let (gx, gy, gz) = if tile_zswap(k) {
                (gc * gw, gh, 1u32)
            } else if chan_x {
                (gc, gh, gw)
            } else {
                (gw, gh, gc)
            };
            eprintln!(
                "launch {k} c_in {} c_out {} h {} w {} oc {} ng {} chan_x {} grid {gx}x{gy}x{gz}",
                x.c, c_out, x.h, x.w, oc, ng, chan_x
            );
        }
        let l = Launch::new(
            if tile_zswap(k) {
                (gc * gw, gh, 1)
            } else if chan_x {
                (gc, gh, gw)
            } else {
                (gw, gh, gc)
            },
            (tbx as u32, tby as u32, ng),
        );
        ca.run(k, l, |a| {
            a.ptr(x.ptr());
            a.ptr(w);
            a.ptr(bias.unwrap_or(0));
            a.ptr(y.ptr());
            a.i32(x.c as i32);
            a.i32(c_out as i32);
            a.i32(x.h as i32);
            a.i32(x.w as i32);
            a.i32(act);
            a.f32(act_p);
        })?;
        return Ok(y);
    }
    if conv_variant() == "tile" {
        // 128 output columns, 8 rows and 8 output channels per block, and the
        // Z dimension carries the output channels, exactly as the kernel's own
        // geometry comment says. The kernel bounds-checks against h/wd itself,
        // so a partially covered tile is fine.
        let l = Launch::new(
            (
                ((x.w + 127) / 128).max(1) as u32,
                ((x.h + 7) / 8).max(1) as u32,
                ((c_out + 7) / 8).max(1) as u32,
            ),
            (32, 8, 1),
        );
        ca.run("lg_conv3x3_tile", l, |a| {
            a.ptr(x.ptr());
            a.ptr(w);
            a.ptr(bias.unwrap_or(0));
            a.ptr(y.ptr());
            a.i32(x.c as i32);
            a.i32(c_out as i32);
            a.i32(x.h as i32);
            a.i32(x.w as i32);
            a.i32(act);
            a.f32(act_p);
        })?;
        return Ok(y);
    }
    ca.run("lg_conv3x3s1p1", conv_grid(c_out, x.h, x.w), |a| {
        a.ptr(x.ptr());
        a.ptr(w);
        a.ptr(bias.unwrap_or(0));
        a.ptr(y.ptr());
        a.i32(x.c as i32);
        a.i32(c_out as i32);
        a.i32(x.h as i32);
        a.i32(x.w as i32);
    })?;
    if act == 2 {
        lrelu_in_place(ca, &y, act_p)?;
    } else if act == 1 {
        relu_in_place(ca, &y)?;
    }
    Ok(y)
}

/// A 3x3 over a concatenation of planes, WITHOUT building the concatenation.
///
/// `src` holds up to FIVE planes in channel order (earlier planes first), `c_out`
/// is the output channel count and `c_in` the SUM of the source channel counts.
/// The kernel's staging loop resolves a concatenated channel index into
/// (plane, channel) with a chain of comparisons, so the dense block no longer
/// has to materialise `cat([x, h1, h2, ...])` with `lg_copy` calls first - 15
/// copies per dense block, 11% of the LR64 run and 376 ms of 3091 at LR256.
///
/// The copies were not otherwise free either: each one wrote a full plane that
/// the conv then read back, so removing them takes both the launch and the
/// bandwidth. Validation is by SUM plus per-plane spatial check rather than by
/// `x.c`, because there is no single `x` here - a mismatch means the caller has
/// miscounted the concatenation and the kernel would read past a plane.
pub fn conv3x3_cat(
    ca: &Cuda,
    src: &[&Plane],
    w: CUdeviceptr,
    bias: Option<CUdeviceptr>,
    c_out: usize,
    act: i32,
    act_p: f32,
) -> Result<Plane, String> {
    let (k, ng) = cat_kernel(src, c_out);
    conv3x3_cat_kernel(ca, src, k, ng, w, bias, c_out, act, act_p)
}

/// Which concatenated kernel a shape gets: `(kernel, NG)`.
///
/// Separated from the launch so that the harness can run a NAMED kernel on the
/// same data as another - the point of the NG comparison being that the two
/// kernels agree, not that a particular shape reaches one of them.
fn cat_kernel(src: &[&Plane], c_out: usize) -> (&'static str, u32) {
    let h = src[0].h;
    let wd = src[0].w;
    let blocks = (((wd + 63) / 64).max(1)) * (((h + 7) / 8).max(1)) * ((c_out + 7) / 8).max(1);
    let gc = ((c_out + 7) / 8).max(1);
    let small = (((wd + 63) / 64).max(1)) * (((h + 7) / 8).max(1)) < 16;
    // NG = 2 folds two output-channel groups into one block so they share the
    // staged input tile, which is the only lever on the input re-read that does
    // not cost registers - see the NG block comment in cuda/hcflow.cu. Two
    // conditions, both REQUIRED:
    //   * the channel-group count must be a whole multiple of NG, or the tail
    //     group would run with idle z-slabs (correct, but the point of the
    //     variant is the amortized staging, so it is not worth having);
    //   * the SMALL-PLANE tile must not be in play: at 64x64 the launch is
    //     already short of blocks (32-64 on 20 SMs) and NG divides the block
    //     count by NG, which is exactly what `catt2s` exists to avoid. The two
    //     are alternatives, not a pair.
    // MEASURED, one process, five shapes: the concatenated conv goes 6.12-6.30 ->
    // 5.07-5.10 ms at 64->64@512x512, 3.09-3.15 -> 2.66 at 64->32, 17.96-18.09 ->
    // 14.94 at 192->64, 9.15-9.29 -> 7.65 at 128->42 and 0.58 -> 0.39 at
    // 128->32@128x128 (1.17-1.50x), with every NG row agreeing with the CPU twin to
    // 1.7e-6. NG = 4 was measured too and is slower everywhere (5.61 against 5.16
    // at 64->64) because the staged WEIGHT tile, unlike the input tile, grows with
    // NG, so only NG = 2 is kept.
    let ng = if small || gc % 2 != 0 || (blocks / 2) < 60 { 1 } else { 2 };
    let k = if small {
        "lg_conv3x3_catt2s"
    } else if ng == 2 {
        // The channel group stays on blockIdx.x (`catq0`), which at NG = 1 leads
        // `catq` by 8-11% and is the same pairing at NG = 2.
        "lg_conv3x3_catq0ng2"
    } else if blocks >= 160 {
        "lg_conv3x3_catq0"
    } else {
        "lg_conv3x3_catx0"
    };
    (k, ng)
}

/// Launch one NAMED concatenated kernel with an explicit NG.
fn conv3x3_cat_kernel(
    ca: &Cuda,
    src: &[&Plane],
    kern: &'static str,
    ng: u32,
    w: CUdeviceptr,
    bias: Option<CUdeviceptr>,
    c_out: usize,
    act: i32,
    act_p: f32,
) -> Result<Plane, String> {
    if src.is_empty() || src.len() > 5 {
        return Err(format!("conv3x3_cat: {} source planes (1..=5)", src.len()));
    }
    let h = src[0].h;
    let wd = src[0].w;
    let c_in: usize = src.iter().map(|p| p.c).sum();
    for p in src {
        if p.h != h || p.w != wd {
            return Err(format!(
                "conv3x3_cat: source planes differ in spatial size ({}x{} vs {}x{})",
                p.h, p.w, h, wd
            ));
        }
    }
    cuda::tag("conv:y");
    trace_shape("cat", c_in, c_out, h, wd);
    let y = Plane::alloc(c_out, h, wd)?;
    let gx = ((wd + 63) / 64).max(1) as u32;
    let gh = ((h + 7) / 8).max(1) as u32;
    let gc = ((c_out + 7) / 8).max(1) as u32;
    let blocks = (((wd + 63) / 64).max(1)) * (((h + 7) / 8).max(1)) * ((c_out + 7) / 8).max(1);
    // The channel group on blockIdx.x: the blocks that share an input tile run
    // consecutively, so the re-read of that tile hits L2 - measured 8-11% on the
    // trunk shapes against the blockIdx.z order. The arithmetic and the
    // accumulation order are unchanged, so `--prim-test`'s splits cover it.
    let chanx = kern != "lg_conv3x3_catx" && kern != "lg_conv3x3_catt2s";
    // REACHABILITY, not logging: a branch that no run takes is indistinguishable
    // from a branch that does not work, so `HCFLOW_SHAPES=1` prints it and the
    // visible side effect is the proof that the dispatch actually fired.
    if ng > 1 && std::env::var("HCFLOW_SHAPES").is_ok() {
        eprintln!(
            "cat-ng: {}x{} c_in {} c_out {} gc {} blocks {} -> NG {}",
            h, wd, c_in, c_out, gc, blocks, ng
        );
    }
    // The grid each kernel was compiled for: `chanx` swaps which of x and z
    // carries the channel group, and `catt2s` is a 32-column, 4-row tile.
    let (l, b) = if kern == "lg_conv3x3_catt2s" {
        (
            Launch::new(
                (
                    ((wd + 63) / 64).max(1) as u32,
                    ((h + 3) / 4).max(1) as u32,
                    gc,
                ),
                (32, 4, 1),
            ),
            (32u32, 4u32),
        )
    } else {
        (
            // NG divides the grid's channel dimension and becomes the block's z
            // axis; the `gc % ng == 0` check above is what makes `gc / ng` exact.
            Launch::new(
                if chanx { (gc / ng, gh, gx) } else { (gx, gh, gc / ng) },
                (16, 8, ng),
            ),
            (16u32, 8u32),
        )
    };
    let _ = b;
    ca.run(kern, l, |a| {
        a.ptr(src[0].ptr());
        a.ptr(w);
        a.ptr(bias.unwrap_or(0));
        a.ptr(y.ptr());
        a.i32(c_in as i32);
        a.i32(c_out as i32);
        a.i32(h as i32);
        a.i32(wd as i32);
        a.i32(act);
        a.f32(act_p);
        // The remaining planes and their channel starts: `gN` is the
        // concatenated index at which plane N begins, so plane N holds channels
        // `[gN, g(N+1))`. A missing plane repeats the last start, which keeps
        // every start non-decreasing - the kernel's chain of comparisons relies
        // on that.
        let mut g = [0i32; 5];
        let mut run = 0usize;
        for (i, s) in src.iter().enumerate() {
            if i > 0 {
                g[i - 1] = run as i32;
            }
            run += s.c;
        }
        for i in src.len().saturating_sub(1)..5 {
            g[i] = run as i32;
        }
        for i in 1..6 {
            let ptr = if i < src.len() { src[i].ptr() } else { src[0].ptr() };
            a.ptr(ptr);
        }
        for i in 0..5 {
            a.i32(g[i]);
        }
        a.i32(src.len() as i32);
    })?;
    Ok(y)
}

/// 1x1. `w` is [c_out][c_in].
pub fn conv1x1(
    ca: &Cuda,
    x: &Plane,
    w: CUdeviceptr,
    bias: Option<CUdeviceptr>,
    c_out: usize,
) -> Result<Plane, String> {
    let y = Plane::alloc(c_out, x.h, x.w)?;
    // The register-blocked 1x1 for anything big enough to fill its 64x64 tiles,
    // the elementwise one for the small planes (its tiles would be mostly
    // padding there). Both fold the bias into the accumulator before the first
    // multiply and walk `c_in` ascending, so this switch is a performance choice
    // and not an arithmetic one: `--prim-test`'s 1x1 case covers the pair.
    let rb = x.h * x.w >= 256 && c_out >= 16;
    let grid = if rb {
        Launch::new(
            (
                ((x.hw() + 63) / 64).max(1) as u32,
                ((c_out + 63) / 64).max(1) as u32,
                1,
            ),
            (16, 16, 1),
        )
    } else {
        conv_grid(c_out, x.h, x.w)
    };
    ca.run(if rb { "lg_conv1x1_rb" } else { "lg_conv1x1" }, grid, |a| {
        a.ptr(x.ptr());
        a.ptr(w);
        a.ptr(bias.unwrap_or(0));
        a.ptr(y.ptr());
        a.i32(x.c as i32);
        a.i32(c_out as i32);
        a.i32(x.h as i32);
        a.i32(x.w as i32);
    })?;
    Ok(y)
}

/// `y = x * scale[c] + shift[c]`, either vector nullable.
pub fn affine(
    ca: &Cuda,
    x: &Plane,
    scale: Option<CUdeviceptr>,
    shift: Option<CUdeviceptr>,
) -> Result<Plane, String> {
    let y = Plane::alloc(x.c, x.h, x.w)?;
    channel_affine(ca, x.ptr(), scale, shift, y.ptr(), x.c, x.hw())?;
    Ok(y)
}

/// ReLU, in place.
pub fn relu_in_place(ca: &Cuda, x: &Plane) -> Result<(), String> {
    ca.run_n("lg_relu", x.numel(), |a| {
        a.ptr(x.ptr());
        a.ptr(x.ptr());
        a.i64(x.numel() as i64);
    })
}

/// Leaky ReLU, in place.
pub fn lrelu_in_place(ca: &Cuda, x: &Plane, slope: f32) -> Result<(), String> {
    ca.run_n("lg_lrelu", x.numel(), |a| {
        a.ptr(x.ptr());
        a.ptr(x.ptr());
        a.f32(slope);
        a.i64(x.numel() as i64);
    })
}

/// Nearest-neighbour 2x upsample.
///
/// The kernel derives `oh = h * 2, ow = wd * 2` from its INPUT geometry and
/// bounds-checks each thread against those, so the grid has to cover the OUTPUT:
/// launching it over the input's geometry writes only the top-left quadrant and
/// leaves the rest of the buffer holding whatever `cuMemAlloc` returned.
pub fn upsample2x(ca: &Cuda, x: &Plane) -> Result<Plane, String> {
    let y = Plane::alloc(x.c, x.h * 2, x.w * 2)?;
    // THE TOOLKIT KERNEL IS THE FAST ONE, and a flow-direct rewrite of it is not
    // worth its complexity: an interleaved A/B of the two (same process, twelve
    // geometries, min of three) has the toolkit's version ahead by 2.4-4.8x on
    // every plane this graph produces - 0.0357 ms against 0.0994 ms at this
    // engine's own 24x128x128 - and level at the penguin-sized ones. Both are
    // exact against each other, so the choice is purely the clock.
    ca.run("lg_upsample2x_nearest", pix_grid(x.c, y.h, y.w), |a| {
        a.ptr(x.ptr());
        a.ptr(y.ptr());
        a.i32(x.c as i32);
        a.i32(x.h as i32);
        a.i32(x.w as i32);
    })?;
    Ok(y)
}

/// Pixel unshuffle, the Squeeze2d inverse: [c][h][w] -> [4c][h/2][w/2].
pub fn pixel_unshuffle2(ca: &Cuda, x: &Plane) -> Result<Plane, String> {
    if x.h % 2 != 0 || x.w % 2 != 0 {
        return Err(format!("pixel_unshuffle2 needs even spatial sizes, got {}x{}", x.h, x.w));
    }
    let y = Plane::alloc(x.c * 4, x.h / 2, x.w / 2)?;
    ca.run("lg_pixel_unshuffle2", pix_grid(x.c, x.h / 2, x.w / 2), |a| {
        a.ptr(x.ptr());
        a.ptr(y.ptr());
        a.i32(x.c as i32);
        a.i32(x.h as i32);
        a.i32(x.w as i32);
    })?;
    Ok(y)
}

/// Unsqueeze2d: [4c][h][w] -> [c][2h][2w]. This is `nn.PixelShuffle(2)`, i.e. the
/// inverse of the `SqueezeLayer` above, and it is the TOOLKIT's kernel:
/// `lg_pixel_shuffle(src, dst, c, h, w, r)` with `r = 2`.
///
/// This engine used to carry its own copy of the permutation, on the strength of a
/// comment in `cuda/hcflow.cu` claiming the toolkit had no pixel-shuffle kernel.
/// That stopped being true when one was promoted, so the two forms were measured
/// against each other, interleaved in ONE process, over sixteen geometries - both
/// real call sites first. Both write byte-identical output at every one of them.
///
/// AT THIS FUNCTION'S TWO CALL SITES THE TWO ARE A WASH: level 0 (6 output
/// channels, 64x64 source) is 4.1 us here against 4.3 us for the project's form,
/// and level 1 (3 channels, 128x128 source) is 6.6 against 6.0 us. That is 0.4 us
/// of a 310 ms pass at LR64, i.e. nothing either way, so the choice is NOT made
/// on speed at the call sites. It is made on what happens away from them: the
/// project's form collapses on the flatter, wider shapes - 0.40x at 128x32x32,
/// 0.61x at 64x32x32, 0.87-0.91x from 24x64x64 up - because it loops over channels
/// inside the thread instead of putting the channel on the grid, and the toolkit's
/// kernel is the only one of the two that does r = 3 and r = n at all. One kernel
/// that is never the slow one beats one duplicate that is sometimes 2.5x slower.
///
/// THE `c` ARGUMENT IS THE OUTPUT CHANNEL COUNT, `x.c / 4` - the kernel writes
/// `c` output planes and reads input channel `ch * r * r + dy * r + dx`, so `c`
/// counts the planes it writes, not the ones it reads. Passing `x.c` here is a 4x
/// write overrun: it faults at 128x128 and at 64x64 lands in still-mapped pool
/// memory and silently corrupts nothing that the next launch reads. The CPU twin
/// (`net::unsqueeze2d`) takes the same `oc = x.c / 4`, which is the form to
/// compare the launch against.
///
/// GEOMETRY: `pix_grid(oc, 2h, 2w)` - `ceil(ow/32)` by `ceil(oh/8)` by `oc`, block
/// (32, 8, 1), with the channel as the SLOWEST grid axis. That is the toolkit
/// kernel's own shape (one block per (32, 8) output patch per channel) and the same
/// shape this engine's other per-pixel-per-channel kernels use, so the existing
/// `pix_grid` helper is already the right launch. `r` is the LAST kernel argument,
/// after the three sizes.
pub fn unsqueeze2d(ca: &Cuda, x: &Plane) -> Result<Plane, String> {
    if x.c % 4 != 0 {
        return Err(format!("unsqueeze2d needs a channel count divisible by 4, got {}", x.c));
    }
    let oc = x.c / 4;
    let y = Plane::alloc(oc, x.h * 2, x.w * 2)?;
    ca.run("lg_pixel_shuffle", pix_grid(oc, y.h, y.w), |a| {
        a.ptr(x.ptr());
        a.ptr(y.ptr());
        a.i32(oc as i32);
        a.i32(x.h as i32);
        a.i32(x.w as i32);
        a.i32(2);
    })?;
    Ok(y)
}

/// `out = base + factor * x`, on the device. Used with `base == out` for the
/// in-place residual and with a scratch `out` otherwise.
fn add_scaled_into(ca: &Cuda, base: &Plane, x: &Plane, out: &Plane, factor: f32) -> Result<(), String> {
    ca.run_n("lg_add_scaled", out.numel(), |a| {
        a.ptr(base.ptr());
        a.ptr(x.ptr());
        a.ptr(out.ptr());
        a.i64(out.numel() as i64);
        a.f32(factor);
    })
}

/// `dst += src`, device to device, same length.
fn copy_into(ca: &Cuda, src: &Plane, dst: &Plane, dst_channel0: usize) -> Result<(), String> {
    if src.numel() == 0 {
        return Ok(());
    }
    ca.run_n("lg_copy", src.numel(), |a| {
        a.ptr(src.ptr());
        // `dst_channel0 * hw` elements past the start of `dst`.
        a.ptr(dst.ptr() + 4 * (dst_channel0 * src.hw()) as u64);
        a.i64(src.numel() as i64);
    })
}

/// Concatenate two planes of equal spatial size along the channel axis.
pub fn cat(ca: &Cuda, a: &Plane, b: &Plane) -> Result<Plane, String> {
    if a.h != b.h || a.w != b.w {
        return Err(format!(
            "cat: {}x{}x{} vs {}x{}x{}",
            a.c, a.h, a.w, b.c, b.h, b.w
        ));
    }
    let y = Plane::alloc(a.c + b.c, a.h, a.w)?;
    copy_into(ca, a, &y, 0)?;
    copy_into(ca, b, &y, a.c)?;
    Ok(y)
}

/// `cat([x, f0, f1, ...])` along channels - the dense block's growing input.
fn concat_all(ca: &Cuda, x: &Plane, feats: &[Plane]) -> Result<Plane, String> {
    let total: usize = x.c + feats.iter().map(|f| f.c).sum::<usize>();
    let y = Plane::alloc(total, x.h, x.w)?;
    copy_into(ca, x, &y, 0)?;
    let mut off = x.c;
    for f in feats {
        if f.h != x.h || f.w != x.w {
            return Err("concat_all: spatial mismatch".to_string());
        }
        copy_into(ca, f, &y, off)?;
        off += f.c;
    }
    Ok(y)
}

/// A plane's channel range, `[c0, c1)`, as a fresh plane.
fn channel_range(ca: &Cuda, x: &Plane, c0: usize, c1: usize) -> Result<Plane, String> {
    if c1 > x.c || c0 > c1 {
        return Err(format!("channel_range: {c0}..{c1} of {} channels", x.c));
    }
    let y = Plane::alloc(c1 - c0, x.h, x.w)?;
    if c1 > c0 {
        let hw = x.hw();
        ca.run_n("lg_copy", (c1 - c0) * hw, |a| {
            a.ptr(x.ptr() + 4 * (c0 * hw) as u64);
            a.ptr(y.ptr());
            a.i64(((c1 - c0) * hw) as i64);
        })?;
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

/// A borrowed weight: where it is on the device and how long it is.
#[derive(Clone, Copy)]
pub struct W {
    pub ptr: CUdeviceptr,
}

/// Everything one flow step needs, already located in the arena. Mirrors
/// `net::Step`, and every length is a length `weights.rs` already validated at
/// load, so a mismatch here means the graph and the checker disagree.
pub struct Step {
    pub an_scale: W,
    pub an_bias: W,
    pub inv_w: W,
    pub c1w: W,
    pub c1b: W,
    pub c2w: W,
    pub c2b: W,
    pub c3w: W,
    pub c3b: W,
    pub hidden: usize,
    /// `z1`'s width: `n / 2` (floor), so an odd `n` puts the extra channel in
    /// `z2` and the coupling's output is `2 * (n - n / 2)`.
    pub half: usize,
    pub out_ch: usize,
}

impl Step {
    /// Channels of the pair the coupling's last conv emits.
    pub fn coupling_out(&self) -> usize {
        2 * (self.out_ch - self.half)
    }
}

/// Fetch a step's tensors from the arena. `n` is the step's `z` width and may
/// be odd; `cond` is the conditioning width added to the coupling's first half.
pub fn load_step(ca: &Cuda, base: &str, n: usize, cond: usize) -> Result<Step, String> {
    let half = Config::half(n);
    let fcn_in = half + cond;
    let hidden = 64;
    let g = |suffix: &str, want: usize| -> Result<W, String> {
        let name = format!("{base}.{suffix}");
        Ok(W { ptr: ca.w(&name, want)? })
    };
    let s = Step {
        an_scale: g("an_scale", n)?,
        an_bias: g("an_bias", n)?,
        inv_w: g("w_rev", n * n)?,
        c1w: g("c1w", hidden * fcn_in * 9)?,
        c1b: g("c1b", hidden)?,
        c2w: g("c2w", hidden * hidden)?,
        c2b: g("c2b", hidden)?,
        c3w: g("c3w", 2 * (n - half) * hidden * 9)?,
        c3b: g("c3b", 2 * (n - half))?,
        hidden,
        half,
        out_ch: n,
    };
    Ok(s)
}

// ---------------------------------------------------------------------------
// the graph
// ---------------------------------------------------------------------------

/// The coupling, reversed, with the shift/scale split and the inversion fused
/// into `lg_couple_inverse` (this project's kernel).
///
/// `z` is [n][h][w]; `z1 = z[0:half]` is passed through and `z2 = z[half:n]` is
/// updated as `z2 * exp(-logscale) - shift`, where `h`'s even channels are the
/// shift and its odd channels are the scale. The FCN reads `cat([z1, cond])`
/// exactly as the reference does.
pub fn coupling_rev(
    ca: &Cuda,
    s: &Step,
    z: &Plane,
    cond: Option<&Plane>,
    out: &mut Plane,
) -> Result<(), String> {
    let z1 = channel_range(ca, z, 0, s.half)?;
    let h_in = match cond {
        Some(c) => cat(ca, &z1, c)?,
        None => channel_range(ca, z, 0, s.half)?,
    };
    let h = conv3x3(ca, &h_in, s.c1w.ptr, Some(s.c1b.ptr), s.hidden)?;
    relu_in_place(ca, &h)?;
    let h2 = conv1x1(ca, &h, s.c2w.ptr, Some(s.c2b.ptr), s.hidden)?;
    drop(h);
    relu_in_place(ca, &h2)?;
    let h3 = conv3x3(ca, &h2, s.c3w.ptr, Some(s.c3b.ptr), s.coupling_out())?;
    drop(h2);

    // The lower half of `out` is `z1` verbatim; the kernel only writes the
    // upper half, so the copy has to happen first.
    copy_into(ca, &z1, out, 0)?;
    let hw = z.hw();
    let gw = ((hw as u64) + 255) / 256;
    let n2 = s.out_ch - s.half;
    // `z` is the STEP's input and `out` the step's output: the two are different
    // buffers, so the kernel takes both. Reading `out`'s second half here would
    // read the uninitialised half of a freshly allocated plane.
    ca.run("lg_couple_inverse", Launch::new((gw.max(1) as u32, n2 as u32, 1), (256, 1, 1)), |a| {
        a.ptr(h3.ptr());
        a.ptr(z.ptr());
        a.ptr(out.ptr());
        a.i32(s.half as i32);
        a.i32(hw as i32);
    })?;
    Ok(())
}

/// One flow step, reversed: coupling -> the inverse 1x1 -> the reversed
/// ActNorm.
///
/// The order is the reference's `FlowStep.reverse`, and it is the exact reverse
/// of forward (ActNorm, then the 1x1, then the coupling). The ActNorm and the
/// 1x1 could be fused into one affine of the 1x1's weight, but they are kept
/// separate so each stage can be compared with the CPU backend on its own - and
/// because the ActNorm is what carries the `-bias` that the fold would have to
/// push through the 1x1.
pub fn flow_step_rev(ca: &Cuda, s: &Step, z: &Plane, cond: Option<&Plane>) -> Result<Plane, String> {
    let mut c = Plane::alloc(z.c, z.h, z.w)?;
    coupling_rev(ca, s, z, cond, &mut c)?;
    let y = conv1x1(ca, &c, s.inv_w.ptr, None, s.out_ch)?;
    drop(c);
    affine(ca, &y, Some(s.an_scale.ptr), Some(s.an_bias.ptr))
}

/// Leaky-ReLU(0.2) then the dense concat, repeated five times: one
/// ResidualDenseBlock.
///
/// With a hidden width nf and a growth channel gc the convs read
/// nf, nf+gc, nf+2gc, nf+3gc and nf+4gc channels and only the last returns to
/// nf. The concat is built with device copies, not a host round trip: at
/// 512x512 there are 15 of them per dense block and a round trip each would
/// dominate the whole trunk.
pub fn rdb(ca: &Cuda, x: &Plane, w: &[W], b: &[W], nf: usize, gc: usize) -> Result<Plane, String> {
    // The dense block's input is `cat([x, h1, h2, ...])`, and materialising it
    // costs one `lg_copy` per input plane per conv - 15 launches per dense block,
    // 11% of the LR64 run and 376 ms of 3091 at LR256 by the profiler. The
    // concatenated tensor is exactly the union of the planes already in hand, so
    // the copies are bought only to make the staged input contiguous; the staged
    // input of the tile kernel is now able to gather across planes instead (see
    // `conv3x3_cat`), which is what makes them removable.
    //
    // Only for the tiled variants: the cat kernels share the tile body but not
    // the other variants' grids, so a `tile` or `direct` run must keep its own
    // path rather than silently launch a different kernel.
    let cat_ok = matches!(conv_variant(), "x16" | "q2" | "dbuf" | "dbuf2");
    let mut feats: Vec<Plane> = Vec::with_capacity(4);
    let mut last: Option<Plane> = None;
    for i in 0..5 {
        let outc = if i < 4 { gc } else { nf };
        // The lrelu(0.2) that follows conv1..conv4 rides in the conv's epilogue
        // on the tiled kernel; the last conv has no activation.
        let act = if i < 4 { 2 } else { 0 };
        let h = if feats.is_empty() || !cat_ok {
            let cat_in = concat_all(ca, x, &feats)?;
            let h = conv3x3_act(ca, &cat_in, w[i].ptr, Some(b[i].ptr), outc, act, 0.2)?;
            drop(cat_in);
            h
        } else {
            let mut planes: Vec<&Plane> = Vec::with_capacity(1 + feats.len());
            planes.push(x);
            for f in &feats {
                planes.push(f);
            }
            conv3x3_cat(ca, &planes, w[i].ptr, Some(b[i].ptr), outc, act, 0.2)?
        };
        if i < 4 {
            feats.push(h);
        } else {
            last = Some(h);
        }
    }
    // The dense stack ends with its OWN residual: `ResidualDenseBlock.forward`
    // is `x5 * 0.2 + x`, and the enclosing RRDB adds a second one. Dropping this
    // one leaves the trunk an amplifier rather than a scale-preserving feature
    // extractor - the same defect the CPU twin had, in the same place.
    let h = last.unwrap();
    let y = Plane::alloc(x.c, x.h, x.w)?;
    copy_into(ca, x, &y, 0)?;
    add_scaled_into(ca, &y, &h, &y, 0.2)?;
    drop(feats);
    drop(h);
    dbg_live("end rdb");
    Ok(y)
}

/// One RRDB: three dense blocks in series with the outer residual
/// `out * 0.2 + x`. Each dense block carries its own residual as well (see
/// `rdb`); there is no ActNorm anywhere in the trunk.
pub fn rrdb(ca: &Cuda, x: &Plane, w: &[W], b: &[W], nf: usize, gc: usize) -> Result<Plane, String> {
    let mut h = Plane::alloc(x.c, x.h, x.w)?;
    copy_into(ca, x, &h, 0)?;
    for r in 0..3 {
        let next = rdb(ca, &h, &w[r * 5..r * 5 + 5], &b[r * 5..r * 5 + 5], nf, gc)?;
        h = next;
    }
    // out = x + 0.2 * h, computed straight into a copy of `x`.
    let y = Plane::alloc(x.c, x.h, x.w)?;
    copy_into(ca, x, &y, 0)?;
    add_scaled_into(ca, &y, &h, &y, 0.2)?;
    drop(h);
    dbg_live("end rrdb");
    Ok(y)
}

/// One trunk: `w.config.rrdb_blocks` RRDBs in series, with every weight pulled
/// out of the arena once.
pub fn trunk(ca: &Cuda, x: &Plane, w: &Weights, level: usize, t: usize, nf: usize, gc: usize)
    -> Result<Plane, String>
{
    let mut cur = Plane::alloc(x.c, x.h, x.w)?;
    copy_into(ca, x, &cur, 0)?;
    for blk in 0..w.config.rrdb_blocks {
        let mut wv: Vec<W> = Vec::with_capacity(15);
        let mut bv: Vec<W> = Vec::with_capacity(15);
        for r in 0..3 {
            let base = rdb_name(&format!("l{level}.trunk{t}"), blk, r);
            for c in 1..=5 {
                let wn = format!("{base}.w{c}");
                let bn = format!("{base}.b{c}");
                // The length comes from the container's own header, and `ca.w`
                // re-checks it, so a mis-sized tensor fails here rather than
                // reading past the weight it was meant to be.
                let wl: usize = w.shape(&wn)?.iter().product();
                let bl: usize = w.shape(&bn)?.iter().product();
                wv.push(W { ptr: ca.w(&wn, wl)? });
                bv.push(W { ptr: ca.w(&bn, bl)? });
            }
        }
        cur = rrdb(ca, &cur, &wv, &bv, nf, gc)?;
    }
    Ok(cur)
}

/// The two trunk outputs of the conditional feature, before the cat.
///
/// `get_conditional_feature_SR`: `u_feature_first = conv_first(u)`,
/// `u_feature1 = trunk0(u_feature_first)`,
/// `u_feature2 = trunk_conv1(trunk1(u_feature1)) + u_feature_first`.
pub fn cond_feature_sr(ca: &Cuda, w: &Weights, level: usize, u: &Plane)
    -> Result<(Plane, Plane), String>
{
    let nf = w.config.hidden;
    let gc = w.config.gc;
    let b = format!("l{level}");
    let f_first = conv3x3(
        ca,
        u,
        ca.w(&format!("{b}.cond_conv_first_w"), nf * u.c * 9)?,
        Some(ca.w(&format!("{b}.cond_conv_first_b"), nf)?),
        nf,
    )?;
    sig(ca, &f_first, &format!("L{level} cond f_first"))?;
    let f1 = trunk(ca, &f_first, w, level, 0, nf, gc)?;
    sig(ca, &f1, &format!("L{level} cond f1"))?;
    let t1 = trunk(ca, &f1, w, level, 1, nf, gc)?;
    let f2 = conv3x3(
        ca,
        &t1,
        ca.w(&format!("{b}.trunk_conv_w"), nf * nf * 9)?,
        Some(ca.w(&format!("{b}.trunk_conv_b"), nf)?),
        nf,
    )?;
    sig(ca, &t1, &format!("L{level} cond t1"))?;
    drop(t1);
    add_scaled_into(ca, &f2, &f_first, &f2, 1.0)?;
    sig(ca, &f2, &format!("L{level} cond f2"))?;
    Ok((f1, f2))
}

/// The 128-channel feature the conditional steps and `f` consume.
pub fn cond_feature128(ca: &Cuda, w: &Weights, level: usize, u: &Plane) -> Result<Plane, String> {
    let (f1, f2) = cond_feature_sr(ca, w, level, u)?;
    cat(ca, &f1, &f2)
}

/// The same, also returning the level's 128-channel conditional feature, which
/// the next level conditions on (the reference's `conditional_feature2`).
pub fn cond_flow_feature(
    ca: &Cuda,
    w: &Weights,
    level: usize,
    z_in: &Plane,
    u: &Plane,
    opts: &Options,
    eps: Option<&Plane>,
) -> Result<(Plane, Plane), String> {
    let cfg = &w.config;
    let ci = cfg.cond_in[level];
    let b = format!("l{level}");
    let feature = cond_feature128(ca, w, level, u)?;
    // `f` reads the 128-channel FEATURE (not the 64-channel sticks), so its
    // weight is [2 * cond_in][2 * hidden][3][3] - `weights.rs` checks the same
    // shape at load, and the two must agree because both name the tensor.
    let h = conv3x3(
        ca,
        &feature,
        ca.w(&format!("{b}.cond_out_w"), 2 * ci * cfg.cond_ch[level] * 9)?,
        Some(ca.w(&format!("{b}.cond_out_b"), 2 * ci)?),
        2 * ci,
    )?;

    // z = mean + exp(eps_std * logs) * eps, with `eps` a UNIT normal (the caller
    // supplies it, or it is all zeros) and `mean = h[0::2]`, `logs = h[1::2]`.
    //
    // The reference's own `GaussianDiag.sample` draws `eps ~ N(0, eps_std^2)`
    // and computes `mean + exp(logs) * eps`, which is the same distribution as
    // the unit-eps form used here. The unit-eps form is the one the engine
    // takes because it stays well defined at `eps_std = 0` (where the
    // reference's own comment notes its version "may cause problem") and
    // because it makes a run reproducible from a seed without also having to
    // encode the temperature.
    // A MISSING eps PLANE MUST BECOME A ZERO PLANE, not a null pointer. The
    // sampler kernel takes eps as an OPTIONAL buffer and multiplies it by
    // `exp(eps_std * logs)`; passing NULL for it makes the kernel read device
    // address zero, and at `eps_std = 0` the product would be zero either way -
    // except that a null read is undefined, not zero, so the graph then depends
    // on whatever the driver left in the low page. It is read once per level
    // before the cond feature is built, so the failure is silent and looks like
    // ordinary numerical drift between two runs of the same input: measured,
    // two runs of the same image differed by up to 25/255 in the output PNG.
    //
    // This is why the `None` arm allocates a zero plane instead of passing a
    // null: every `--eps-std 0` run - the mode the reference comparison uses -
    // reaches it.
    let zero_eps;
    let eps = match eps {
        Some(e) => {
            if e.c != ci || e.h != z_in.h || e.w != z_in.w {
                return Err(format!(
                    "level {level}: eps is {}x{}x{}, expected {ci}x{}x{}",
                    e.c, e.h, e.w, z_in.h, z_in.w
                ));
            }
            Some(e.ptr())
        }
        None => {
            zero_eps = Plane::zeros(ci, z_in.h, z_in.w)?;
            Some(zero_eps.ptr())
        }
    };
    let mut z = Plane::alloc(ci, z_in.h, z_in.w)?;
    let hw = z_in.hw();
    let gw = ((hw as u64) + 255) / 256;
    ca.run("lg_sample_latent", Launch::new((gw.max(1) as u32, ci as u32, 1), (256, 1, 1)), |a| {
        a.ptr(h.ptr());
        a.ptr(eps.unwrap_or(0));
        a.ptr(z.ptr());
        a.i32(hw as i32);
        a.f32(opts.eps_std);
    })?;
    drop(h);

    let nsteps = cfg.cond_steps[level];
    for i in (0..nsteps).rev() {
        let s = load_step(ca, &cond_step(&b, i), ci, 2 * cfg.hidden)?;
        z = flow_step_rev(ca, &s, &z, Some(&feature))?;
    }
    Ok((z, feature))
}

/// The reverse of a level's unconditional steps.
pub fn steps_rev(ca: &Cuda, _w: &Weights, level: usize, n: usize, count: usize, z: Plane)
    -> Result<Plane, String>
{
    let b = format!("l{level}");
    let mut z = z;
    for i in (0..count).rev() {
        let s = load_step(ca, &step(&b, i), n, 0)?;
        z = flow_step_rev(ca, &s, &z, None)?;
    }
    Ok(z)
}

/// The whole network on the device, walked exactly as `net::forward_cpu` does.
///
/// `z` starts as the LR image; each level builds its conditional feature, samples
/// the latent the Split consumed, cats it on (the Split inverse), runs thirteen
/// unconditional steps in reverse, and applies Unsqueeze2d. The output is
/// clamped to [0, 1].
pub fn forward_gpu(
    ca: &Cuda,
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
    let mut z = Plane::alloc(lr.c, lr.h, lr.w)?;
    copy_into(ca, lr, &z, 0)?;
    sig(ca, &z, "input")?;
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

        let u = match (&feature_prev, cfg.cond_levels[level]) {
            (Some(f), n) if n > 0 => cat(ca, &z, &upsample2x(ca, f)?)?,
            _ => {
                let c = Plane::alloc(z.c, z.h, z.w)?;
                copy_into(ca, &z, &c, 0)?;
                c
            }
        };

        // The eps planes live on the host (they come from the caller); the
        // device copy is made once per level here rather than once per level
        // twice in the sampler.
        let eps_host = opts.eps.as_ref().and_then(|v| v.get(level));
        let eps_dev = match eps_host {
            Some(p) => Some(Plane::from_host(&p.data, p.c, p.h, p.w)?),
            None => None,
        };
        let (a, feature) = cond_flow_feature(ca, w, level, &z, &u, opts, eps_dev.as_ref())?;
        sig(ca, &feature, &format!("L{level} feature"))?;
        sig(ca, &a, &format!("L{level} sampled a"))?;
        if a.c != ci {
            return Err(format!(
                "level {level}: conditional flow gave {} channels, expected {ci}",
                a.c
            ));
        }
        z = cat(ca, &z, &a)?;
        drop(a);

        z = steps_rev(ca, w, level, full, cfg.steps[level], z)?;
        sig(ca, &z, &format!("L{level} steps_rev"))?;
        z = unsqueeze2d(ca, &z)?;
        sig(ca, &z, &format!("L{level} unsqueeze"))?;
        // The NEXT level conditions on this level's 128-channel FEATURE, not on
        // the conditioning input it was built from - see `net::forward_cpu`.
        feature_prev = Some(feature);
        ca.sync()?;
    }
    progress("unsqueeze");
    let mut y = z;
    if y.c != 3 {
        return Err(format!("output has {} channels, expected 3", y.c));
    }
    // The clamp is the last op of the network and touches only the 3-channel
    // output, so it is done on the host: at 1024x1024 that is 12 MiB in each
    // direction ONCE per image, against a kernel launch per image for a device
    // clamp. `host_clamp` re-uploads in place, so the plane keeps its shape and
    // the caller still gets a device buffer.
    host_clamp(&mut y, 0.0, 1.0)?;
    Ok(y)
}

/// Print a plane's element sum, minimum and maximum, when `HCFLOW_SIG` is set.
///
/// The point of this rather than a diff of the output PNG: the output goes
/// through a host clamp and an 8-bit PNG quantisation, so two runs that differ
/// in the last bit of one pixel can print the same image and two that differ in
/// the 1e-3 place can print different ones. A signature per plane says WHERE two
/// runs first disagree, which is the question that matters when the same command
/// gives two different pictures.
pub fn sig(ca: &Cuda, x: &Plane, tag: &str) -> Result<(), String> {
    if std::env::var("HCFLOW_SIG").is_err() {
        return Ok(());
    }
    ca.sync()?;
    let v = x.to_host()?;
    let sum: f64 = v.iter().map(|a| *a as f64).sum();
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan = v.iter().filter(|a| !a.is_finite()).count();
    eprintln!(
        "SIG {tag:<28} {}{}x{}x{} sum {:.9e} min {:.6e} max {:.6e}{}",
        x.c, if x.c < 10 { " " } else { "" }, x.h, x.w, sum, mn, mx,
        if nan > 0 { format!(" NONFINITE {nan}") } else { String::new() }
    );
    Ok(())
}

/// Clamp every element of `y` into `[lo, hi]`, through the host.
///
/// Used once per image on a 3-channel plane; a device kernel here would save a
/// 12 MiB round trip at 1024x1024 and cost a launch, which is the wrong trade
/// for the last op of the graph - and the CPU twin clamps on the host too, so
/// the two backends compare on identical arithmetic.
fn host_clamp(y: &mut Plane, lo: f32, hi: f32) -> Result<(), String> {
    let mut v = y.to_host()?;
    for e in v.iter_mut() {
        *e = e.clamp(lo, hi);
    }
    y.buf.upload(&v)
}


// ---------------------------------------------------------------------------
// primitive parity: every device op against its CPU twin
// ---------------------------------------------------------------------------

/// A tiny deterministic generator, so both sides see identical inputs without
/// adding a dependency.
fn prand(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

/// Run every primitive here against the same primitive in `net.rs` on the same
/// data and report the largest difference. A graph-level disagreement with a
/// verified CPU backend says one of these is wrong; this says which.
/// The NG (channel-groups-per-block) falsification test: ONE process, the same
/// buffers, alternating rows, correctness and speed together.
///
/// WHY THIS IS A SEPARATE PATH rather than a `--conv-bench` row. Two reasons, both
/// structural. (1) The concatenated conv cannot go through the bench's launcher:
/// `conv3x3_act` pushes the 10-argument list, and a `*cat*` kernel declares 21
/// (five planes, five channel starts, `ng`), so timing one that way is an
/// argument-count mismatch rather than a wrong number - it has to run through
/// `conv3x3_cat`. (2) A bench list writes the grid from its own table, and the
/// question here is not "what does this kernel do at a chosen grid" but "does the
/// GRAPH's dispatch, with its `gc % ng` and `blocks / ng` guards, produce a
/// correct and faster conv". So this exercises the real entry points.
///
/// The A/B is INSIDE one process and the rows ALTERNATE (1, 2, 4, 1, 2, 4): on
/// this machine the same command in another process differs by up to 2.2x, and a
/// bench list has three times failed to predict end-to-end behaviour, so the only
/// reading worth taking is adjacent rows of the same buffer set.
///
/// THE PREDICTION IT TESTS. One call stages `GC * c_in * (TH+2)(TW+2) * 4` bytes
/// of input plus `9 * c_in * c_out * 4 * tiles` of weights, so at NG = 1 the
/// staged traffic is 24.0 FLOP/byte and the measured 3.15-3.20 TFLOP/s is that
/// traffic served at ~135 GB/s. NG = 2 raises it to 43.7 FLOP/byte and NG = 4 to
/// 74.2. If the staging phase is byte-bound rather than per-block-issue-bound,
/// NG = 2 must move the wide shapes to roughly 1.8x the NG = 1 rate. If it does
/// not, this variant is dead and the byte model is wrong.
pub fn ng_selftest(ca: &Cuda, verbose: bool) -> Result<(), String> {
    let shape: Vec<usize> = std::env::var("HCFLOW_NG_SHAPE")
        .unwrap_or_else(|_| "128 32 512 512".to_string())
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if shape.len() != 4 {
        return Err("HCFLOW_NG_SHAPE wants `c_in c_out h w`".to_string());
    }
    let (c_in, c_out, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let iters: usize = std::env::var("HCFLOW_NG_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let rounds: usize = std::env::var("HCFLOW_NG_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let flops = 2.0 * (c_in * c_out * 9 * h * w) as f64;
    println!(
        "ng-test {c_in}->{c_out} at {h}x{w}: {:.1} MFLOP/conv, {iters} iters, {rounds} rounds",
        flops / 1e6
    );

    let mut st = 0x243f_6a88_85a3_08d3u64;
    let xv: Vec<f32> = (0..c_in * h * w).map(|_| prand(&mut st)).collect();
    let wv: Vec<f32> = (0..c_out * c_in * 9).map(|_| prand(&mut st) * 0.1).collect();
    let bv: Vec<f32> = (0..c_out).map(|_| prand(&mut st) * 0.1).collect();
    let x = Plane::from_host(&xv, c_in, h, w)?;
    let wd = cuda::DevBuf::from_host(&wv)?;
    let bd = cuda::DevBuf::from_host(&bv)?;
    // The concatenated form: the same channels, split across SEVERAL planes, which
    // is the shape the graph's dense blocks run (96/128/160->32, 192->64); the
    // kernel reads exactly the bytes a materialised concat would hold.
    //
    // WHICH PLANE LAYOUT, and why it is not always two halves. The graph's dense
    // blocks do NOT split their input in halves: `rdb` builds `cat([x, h1, h2,
    // ...])` from the block input plus its 32-channel growth planes, so the real
    // layouts are 64+32, 64+32+32, 64+32+32+32 and 64+32+32+32+32 - TWO to FIVE
    // planes of UNEQUAL width. Splitting in halves exercises only the two-equal-
    // plane case, which is exactly why this harness reported every NG row correct
    // while the graph's own concatenated conv returned a wrong plane.
    // `HCFLOW_NG_LAYOUT="64,32,32"` selects the graph's layout; the entries must
    // sum to `c_in`.
    let layout: Vec<usize> = std::env::var("HCFLOW_NG_LAYOUT")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect())
        .unwrap_or_default();
    let layout = if layout.is_empty() { vec![c_in / 2, c_in - c_in / 2] } else { layout };
    let total: usize = layout.iter().sum();
    if total != c_in {
        return Err(format!("HCFLOW_NG_LAYOUT {layout:?} sums to {total}, not c_in {c_in}"));
    }
    if layout.len() > 5 {
        return Err(format!("HCFLOW_NG_LAYOUT {layout:?}: at most 5 planes"));
    }
    if verbose {
        eprintln!("  plane layout {layout:?}");
    }
    let mut planes: Vec<Plane> = Vec::with_capacity(layout.len());
    let mut off = 0usize;
    for &c in &layout {
        planes.push(Plane::from_host(&xv[off * h * w..(off + c) * h * w], c, h, w)?);
        off += c;
    }
    let refs: Vec<&Plane> = planes.iter().collect();

    // The CPU twin, from the same values: the concatenation in channel order is
    // planes 1 then 2, so the reference conv is over `xv` itself.
    let cpu_x = net::Plane::from_vec(c_in, h, w, xv.clone());
    let want = net::conv3x3(&cpu_x, &wv, Some(&bv), c_out);

    let cmp = |name: &str, got: &Plane| -> Result<f32, String> {
        let g = got.to_host()?;
        if g.len() != want.data.len() {
            return Err(format!("{name}: {} vs {} elements", g.len(), want.data.len()));
        }
        let d = g.iter().zip(want.data.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let sc = want.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        if d / sc > 1e-4 {
            return Err(format!("{name} disagrees with the CPU twin by {d:.3e} ({:.2e} relative)", d / sc));
        }
        if verbose {
            eprintln!("  {name:<28} max|diff| {d:.3e} (relative {:.2e})", d / sc);
        }
        Ok(d / sc)
    };

    // 1. CORRECTNESS FIRST, and not only at the NG the timing tries: an NG whose
    //    group count does not divide the channel count is still legal (the
    //    kernel's bounds predicates cover a partial group), so a guard that is
    //    wrong shows up here as a wrong plane rather than as a slow one.
    println!("correctness (the concatenated kernels, both NGs):");
    // The two kernels are named directly rather than selected through the graph's
    // guard chain: the chain decides by shape (channel groups, block count), and
    // the point of THIS loop is that the NG = 1 and NG = 2 kernels produce the
    // same plane on the same data, whatever the graph would pick.
    for (kern, ng) in [("lg_conv3x3_catq0", 1u32), ("lg_conv3x3_catq0ng2", 2)] {
        let g = conv3x3_cat_kernel(ca, &refs, kern, ng, wd.ptr, Some(bd.ptr), c_out, 2, 0.2)?;
        // The reference for the fused epilogue.
        let mut e = want.clone();
        net::leaky_relu(&mut e, 0.2);
        let g = g.to_host()?;
        let d = g.iter().zip(e.data.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let sc = e.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        println!("  cat NG={ng}  max|diff| {d:.3e} (relative {:.2e})", d / sc);
        if d / sc > 1e-4 {
            return Err(format!("cat NG={ng} disagrees with the CPU twin ({:.2e} relative)", d / sc));
        }
    }
    // And the plain conv, through the bench's own variant/geometry state, which
    // is also the only way to time it with the grid that geometry implies.
    println!("correctness (plain conv):");
    for (v, ng) in [("q2", 1u32), ("q2ng2", 2)] {
        set_conv_bench_variant(v);
        set_conv_bench_geom(this_projects_tile(v));
        let g = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 0, 0.0)?;
        set_conv_bench_variant("q2");
        set_conv_bench_geom(None);
        println!("  {v} (NG={ng})  rel {:.2e}", cmp(v, &g)?);
    }
    // AND THE GRAPH'S OWN DISPATCH, with the bench state cleared. The rows above
    // pass `this_projects_tile(v)` as the bench geometry, which SKIPS the graph's
    // whole `if conv_bench_geom().is_none()` block - the pick, the q2ng2
    // substitution and the small-tile branch all live in there. A harness that
    // only ever benches with an explicit geometry therefore cannot see a defect in
    // the substitution itself, which is how every NG row here reported agreement
    // while the graph's own concatenated/plain dispatch returned a wrong plane.
    set_conv_bench_variant("q2");
    set_conv_bench_geom(None);
    println!("correctness (the graph's own dispatch, no bench override):");
    // THE ZERO PASS, and it is not optional. Every row above compares against a
    // plane the POOL recycled, so a launch that writes only PART of its output is
    // invisible whenever the block it lands on already holds the right values -
    // which is exactly what happened here: a grid whose channel groups and tile
    // columns were transposed (`chan_x` false against a ZFIRST = 0 template) left
    // half of every plane unwritten, and the row that ran before had written those
    // values correctly, so this harness reported 3.78e-7 while the graph, whose
    // planes carry unrelated data, failed by 7.3e-1. `HCFLOW_ZERO` makes
    // `Plane::alloc` zero the whole plane, so an unwritten element compares
    // against a zero instead of against its own earlier contents and the same row
    // fails at 1.00e0 relative. Run BOTH and require both.
    std::env::set_var("HCFLOW_ZERO", "1");
    let z = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 0, 0.0)?;
    std::env::remove_var("HCFLOW_ZERO");
    let dz = cmp("graph dispatch (zeroed output)", &z)?;
    println!("  graph dispatch, ZEROED output  rel {dz:.2e}");
    if dz > 1e-4 {
        return Err(format!(
            "the graph's own dispatch for {c_in}->{c_out}@{h}x{w} leaves part of \
             its output UNWRITTEN: with the output plane zeroed it disagrees with \
             the CPU twin by {dz:.2e} relative, while the recycled-plane rows \
             above agree. That is a grid/geometry mismatch (which blockIdx axis \
             carries the tile column and which the channel group), not a numerical \
             one."
        ));
    }
    for _ in 0..2 {
        let g = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 0, 0.0)?;
        let d = cmp("graph dispatch", &g)?;
        println!("  graph dispatch  rel {d:.2e}  (kernel not printed here: set HCFLOW_LAUNCH=1)");
        if d > 1e-4 {
            return Err(format!(
                "the graph's own dispatch for {c_in}->{c_out}@{h}x{w}                  disagrees with the CPU twin ({d:.2e} relative) - the bench rows                  above use an explicit geometry and so never exercised it"
            ));
        }
    }

    // 2. SPEED, alternating. Each row is warmed up on its own variant before its
    //    own timing so a JIT or an allocator fault cannot land inside one row
    //    only, and the rows repeat so drift is visible rather than absorbed.
    println!("timings (ms/conv and GFLOP/s):");
    let cat_pick = cat_kernel(&refs, c_out);
    println!(
        "  (the graph's own dispatch for this shape: {} at NG = {})",
        cat_pick.0, cat_pick.1
    );
    for round in 0..rounds {
        for (kern, ng) in [("lg_conv3x3_catq0", 1u32), ("lg_conv3x3_catq0ng2", 2)] {
            for _ in 0..3 {
                let y = conv3x3_cat_kernel(ca, &refs, kern, ng, wd.ptr, Some(bd.ptr), c_out, 2, 0.2)?;
                drop(y);
            }
            ca.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let y = conv3x3_cat_kernel(ca, &refs, kern, ng, wd.ptr, Some(bd.ptr), c_out, 2, 0.2)?;
                drop(y);
            }
            ca.sync()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!(
                "  r{round} cat NG={ng}  {ms:8.3} ms  {:8.1} GFLOP/s",
                flops / (ms / 1e3) / 1e9
            );
        }
        for (v, ng) in [("q2", 1u32), ("q2ng2", 2)] {
            set_conv_bench_variant(v);
            set_conv_bench_geom(this_projects_tile(v));
            for _ in 0..3 {
                let y = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 0, 0.0)?;
                drop(y);
            }
            ca.sync()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let y = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 0, 0.0)?;
                drop(y);
            }
            ca.sync()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!(
                "  r{round} plain {v} (NG={ng})  {ms:8.3} ms  {:8.1} GFLOP/s",
                flops / (ms / 1e3) / 1e9
            );
        }
    }
    set_conv_bench_variant("q2");
    set_conv_bench_geom(None);
    println!("ng-test done");
    Ok(())
}

pub fn prim_selftest(ca: &Cuda, verbose: bool) -> Result<(), String> {
    let mut st = 0x1234_5678_9abc_def0u64;
    let (c_in, c_out, h, w) = (3usize, 8usize, 6usize, 6usize);
    let n = c_in * h * w;
    let xv: Vec<f32> = (0..n).map(|_| prand(&mut st)).collect();
    let wv: Vec<f32> = (0..c_out * c_in * 9).map(|_| prand(&mut st) * 0.3).collect();
    let bv: Vec<f32> = (0..c_out).map(|_| prand(&mut st) * 0.1).collect();
    let w1: Vec<f32> = (0..c_out * c_in).map(|_| prand(&mut st) * 0.4).collect();
    let scale: Vec<f32> = (0..c_in).map(|_| prand(&mut st) + 0.5).collect();
    let shift: Vec<f32> = (0..c_in).map(|_| prand(&mut st) * 0.2).collect();

    let x = Plane::from_host(&xv, c_in, h, w)?;
    let cpu_x = net::Plane::from_vec(c_in, h, w, xv.clone());
    let w_dev = cuda::DevBuf::from_host(&wv)?;
    let b_dev = cuda::DevBuf::from_host(&bv)?;
    let w1_dev = cuda::DevBuf::from_host(&w1)?;
    let sc_dev = cuda::DevBuf::from_host(&scale)?;
    let sh_dev = cuda::DevBuf::from_host(&shift)?;

    let mut worst: Vec<(String, f32)> = Vec::new();
    let mut cmp = |name: &str, got: &Plane, want: &net::Plane| -> Result<(), String> {
        let g = got.to_host()?;
        if g.len() != want.data.len() {
            return Err(format!("{name}: {} vs {} elements", g.len(), want.data.len()));
        }
        let d = g.iter().zip(want.data.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let scale = want.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        if verbose {
            eprintln!("  {name:<22} max|diff| {d:.3e}  (relative {:.2e})", d / scale);
        }
        worst.push((name.to_string(), d / scale));
        Ok(())
    };

    // 3x3 conv, in the variant the graph will use.
    let g = conv3x3(ca, &x, w_dev.ptr, Some(b_dev.ptr), c_out)?;
    cmp(&format!("conv3x3 ({})", conv_variant()), &g, &net::conv3x3(&cpu_x, &wv, Some(&bv), c_out))?;

    // The same conv with the fused epilogue the dense blocks use.
    let g = conv3x3_act(ca, &x, w_dev.ptr, Some(b_dev.ptr), c_out, 2, 0.2)?;
    let mut want = net::conv3x3(&cpu_x, &wv, Some(&bv), c_out);
    net::leaky_relu(&mut want, 0.2);
    cmp("conv3x3+lrelu(0.2)", &g, &want)?;

    // The 3x3 over a CONCATENATION of planes, which is what the dense blocks
    // do without materialising `cat(...)`. Three cases, because the kernel
    // resolves a concatenated channel index through a chain of comparisons and
    // the branch taken depends on how the channels are distributed: an even
    // split of the three channels, a lopsided split, and four planes of one
    // channel each, which is the deepest the chain goes. Every one of them must
    // equal the plain conv over the same channels in the same order.
    for split in [vec![1usize, 1, 1], vec![2, 1], vec![1, 1, 1, 0]].iter() {
        let mut planes: Vec<Plane> = Vec::new();
        let mut at = 0usize;
        for &cn in split.iter() {
            if cn == 0 {
                continue;
            }
            let seg = xv[at * h * w..(at + cn) * h * w].to_vec();
            planes.push(Plane::from_host(&seg, cn, h, w)?);
            at += cn;
        }
        let refs: Vec<&Plane> = planes.iter().collect();
        let g = conv3x3_cat(ca, &refs, w_dev.ptr, Some(b_dev.ptr), c_out, 0, 0.0)?;
        cmp(
            &format!("conv3x3_cat{:?}", split),
            &g,
            &net::conv3x3(&cpu_x, &wv, Some(&bv), c_out),
        )?;
    }

    // 1x1 conv.
    let g = conv1x1(ca, &x, w1_dev.ptr, Some(b_dev.ptr), c_out)?;
    cmp("conv1x1", &g, &net::conv1x1(&cpu_x, &w1, Some(&bv), c_out))?;

    // Per-channel affine.
    let g = affine(ca, &x, Some(sc_dev.ptr), Some(sh_dev.ptr))?;
    let mut cpu_a = cpu_x.clone();
    net::channel_affine(&mut cpu_a, &scale, Some(&shift));
    cmp("channel_affine", &g, &cpu_a)?;

    // Activation (in place on a copy).
    let g = Plane::from_host(&xv, c_in, h, w)?;
    relu_in_place(ca, &g)?;
    let mut cpu_r = cpu_x.clone();
    net::relu(&mut cpu_r);
    cmp("relu", &g, &cpu_r)?;

    // Nearest 2x upsample.
    let g = upsample2x(ca, &x)?;
    cmp("upsample2x", &g, &net::upsample2x(&cpu_x))?;

    // The two permutations.
    let g = pixel_unshuffle2(ca, &x)?;
    cmp("pixel_unshuffle2", &g, &net::pixel_unshuffle2(&cpu_x))?;

    let (c4, h4, w4) = (c_in * 4, h / 2, w / 2);
    let xv4: Vec<f32> = (0..c4 * h4 * w4).map(|_| prand(&mut st)).collect();
    let x4 = Plane::from_host(&xv4, c4, h4, w4)?;
    let cpu_x4 = net::Plane::from_vec(c4, h4, w4, xv4.clone());
    let g = unsqueeze2d(ca, &x4)?;
    cmp("unsqueeze2d", &g, &net::unsqueeze2d(&cpu_x4))?;

    // Concatenation, which is the dense block's inner loop.
    let g = cat(ca, &x, &x)?;
    cmp("cat", &g, &net::cat(&cpu_x, &cpu_x))?;

    // The residual `out = base + factor * x`, used by both residuals.
    let g = Plane::from_host(&xv, c_in, h, w)?;
    add_scaled_into(ca, &g, &x, &g, 0.2)?;
    let want = net::Plane::from_vec(
        c_in,
        h,
        w,
        cpu_x.data.iter().map(|v| v + 0.2 * v).collect(),
    );
    cmp("add_scaled", &g, &want)?;

    let mut worst_sorted = worst.clone();
    worst_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let (name, rel) = &worst_sorted[0];
    println!("primitive parity: worst is `{name}` at relative {rel:.3e}");
    if *rel > 1e-3 {
        return Err(format!("`{name}` disagrees with its CPU twin by {rel:.3e}"));
    }
    println!("all primitives agree with the CPU twin");
    ca.sync()
}

// ---------------------------------------------------------------------------
// conv benchmarking: the three 3x3 variants on identical data
// ---------------------------------------------------------------------------

/// Time one 3x3 conv shape under each variant and report ms and GFLOP/s.
///
/// The graph is 210 dense 3x3 convs per level against 52 coupling convs, so the
/// 3x3 kernel is the whole runtime and the choice between `direct`, `tile` and
/// the tiled family is the whole performance question. Measuring it inside the graph
/// means measuring everything else too; this measures the conv alone, on the
/// same buffers, with the same weights, at the shapes the graph actually runs -
/// which is the only way to tell a slow kernel from a slow launch pattern.
///
/// The weights are deterministic pseudo-random, not the checkpoint's: the timing
/// of a dense conv does not depend on the values, and this keeps the bench
/// runnable without a model file. GFLOP/s counts a multiply-add as two flops and
/// is computed from the DIRECT form's arithmetic - `2 * c_in * c_out * 9 * h * w`
/// - for every variant, so a probe row like `stage` (which does 1/9 of the
/// multiplies) reports the throughput of the OP it implements rather than of the
/// arithmetic it performs.
pub fn conv_bench(
    ca: &Cuda,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    iters: usize,
) -> Result<(), String> {
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let xv: Vec<f32> = (0..c_in * h * w).map(|_| prand(&mut st)).collect();
    let wv: Vec<f32> = (0..c_out * c_in * 9).map(|_| prand(&mut st) * 0.1).collect();
    let bv: Vec<f32> = (0..c_out).map(|_| prand(&mut st) * 0.1).collect();
    let x = Plane::from_host(&xv, c_in, h, w)?;
    let wd = cuda::DevBuf::from_host(&wv)?;
    let bd = cuda::DevBuf::from_host(&bv)?;
    let flops = 2.0 * (c_in * c_out * 9 * h * w) as f64;

    println!(
        "conv3x3 {c_in}->{c_out} at {h}x{w}: {:.1} MFLOP/conv, {iters} iterations",
        flops / 1e6
    );
    // The variant list can be overridden so one process can time a specific A/B
    // - including the SAME variant twice, which is the only way to tell a real
    // difference from drift when two processes disagree by up to 12%.
    let list: Vec<String> = match std::env::var("HCFLOW_BENCH_LIST") {
        Ok(v) if !v.is_empty() => v.split(',').map(|s| s.trim().to_string()).collect(),
        _ => ["x16", "stage", "x16", "q2", "dbuf", "dbuf2"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    for variant in &list {
        let variant = variant.as_str();
        // The variant and its geometry go through the bench's own state rather
        // than the environment: `conv3x3_act` reads an env var at EVERY launch,
        // so a long bench could pick up a change mid-timing, and a variant whose
        // name is not recognised has to be reported rather than silently replaced
        // by the default.
        set_conv_bench_variant(variant);
        let want = variant.to_ascii_lowercase();
        let geom = if let Some(g) = this_projects_tile(&want) {
            Some(g)
        } else if want == "direct" || want == "tile" {
            None
        } else {
            println!("  {variant:<9} (not a variant this project knows - SKIPPED)");
            set_conv_bench_geom(None);
            continue;
        };
        set_conv_bench_geom(geom);
        // Warm up: the first launch of a module pays its own load and the
        // allocator is still handing out fresh pages.
        for _ in 0..3 {
            let y = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 2, 0.2)?;
            drop(y);
        }
        ca.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let y = conv3x3_act(ca, &x, wd.ptr, Some(bd.ptr), c_out, 2, 0.2)?;
            drop(y);
        }
        ca.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "  {variant:<9} {ms:8.3} ms/conv  {:8.1} GFLOP/s",
            flops / (ms / 1e3) / 1e9
        );
    }

    Ok(())
}
