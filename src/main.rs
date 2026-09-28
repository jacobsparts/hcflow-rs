//! HCFlow x4 super-resolution, standalone.
//!
//! Usage:
//!   hcflow --model hcflow_x4.safetensors -i in.png -o out.png
//!   hcflow --model ... -i in.png --device cpu --eps-std 0.9
//!
//! The model file is the output of `tools/convert.py`, not the original .pth.
//!
//! WHAT "eps_std" IS. HCFlow is a conditional normalizing flow: its output is a
//! SAMPLE, drawn as `z = mean + exp(eps_std * logs) * eps` with a unit normal
//! `eps`. `eps_std = 0` makes the map deterministic (the reference's own
//! `--eps_std 0` "mean" mode) and is what a run that wants to be reproducible
//! for a fixed arithmetic should use; the checkpoint's trained value is 0.9.
//! `--seed` makes a non-zero `eps_std` reproducible too, by drawing every
//! level's `eps` from one seeded generator before the walk starts.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod gpu;
mod image;
mod net;
mod weights;

use std::time::Instant;

// Seconds spent in device->host downloads. `--to-host` reports it because the
// default forward time is the GRAPH plus one synchronise with the image
// marshalling outside it - which is the right thing for comparing kernels and
// the wrong thing for comparing against a reference whose number includes its
// own host transfer. A thread-local rather than a global because the run is
// single-threaded and a lock around a timer is a worse trade than the
// restriction.
thread_local! {
    static HOST_SECS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> ! {
    eprintln!(
        "hcflow {VERSION} - HCFlow (ICCV 2021) conditional-flow x4 super-resolution on lightgpu

USAGE:
    hcflow --model <weights.safetensors> -i <in.png> -o <out.png> [options]

OPTIONS:
    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, or - for stdin (default: stdin)
    -o, --output <path>   output PNG, or - for stdout (default: stdout)
        --device <dev>    gpu or cpu (default: gpu when the CUDA driver can be
                          brought up, cpu otherwise; a CPU-only build is always
                          cpu)
        --eps-std <f>     sampling temperature (default: the checkpoint's own,
                          0.9; 0 is deterministic)
        --seed <n>        seed for the latent noise (default: a fresh draw)
        --cpu             same as --device cpu
        --gpu             same as --device gpu, and refuses to fall back
    -q, --quiet           no progress output
        --self-test       run the CPU graph's internal checks and exit
        --trace           walk the CPU graph stage by stage, printing each
                          plane's range (HCFLOW_TRACE_N sets the input size)
        --ref-run <lr.npy> <ref.npy>
                          run the graph on a numpy (1,3,h,w) LR tensor at
                          eps_std 0 and compare with a reference output
        --cuda-selftest   compare the CUDA graph against the CPU graph on one
                          fixed eps draw and exit
        --prim-test       check every device primitive against its CPU twin
                          on the same data and exit (needs --model)
        --ng-test         check the 3x3's NG = 2 substitution against the CPU
                          twin, including once with every output plane zeroed so
                          that a kernel which leaves part of its output unwritten
                          fails instead of reading back the recycled block
        --conv-bench [c_in c_out h w iters]
                          time a 3x3 variant on identical data; no model needed,
                          the data is synthetic (with no numbers it benches the
                          graph's own shapes). HCFLOW_BENCH_LIST=names picks the
                          variants, default `x16,stage,x16,q2,dbuf,dbuf2`
        --parity <n>      run BOTH backends on an n x n input and report the
                          largest difference and their timings
    -h, --help            this text
    -V, --version         print the version

MEMORY:
    A run that cannot fit fails immediately, before allocating, naming the input
    size and the estimate. The estimate is a bounding envelope of measured peaks
    from 64x64 to 512x512 (see `estimated_peak_bytes` in the source), so it is
    generous at the top end; the alternative is a crash several hundred
    allocations into a graph, or a machine that pages for minutes first."
    );
    std::process::exit(2)
}

/// A small deterministic generator for the latent noise.
///
/// The reference draws with `torch.normal`, which the engine has no reason to
/// reproduce bit for bit - what a seed has to give is the SAME noise for the
/// same seed in both backends, and that is what this provides. xorshift64* is
/// used rather than a standard normal table: it is four lines, has no
/// dependencies, and Box-Muller on top of it is exactly as reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// A uniform in (0, 1): the +0.5 keeps `ln` away from zero.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// One standard normal, by Box-Muller.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Draw one `eps` plane per level: `cond_in[level]` channels at that level's
/// resolution.
///
/// The walk RUNS UPWARD in resolution and downward in channels: `z` starts as
/// the 3-channel LR image, the first level's conditional flow samples its latent
/// at that same resolution, then the level's Unsqueeze2d doubles it - so level
/// `i`'s latent lives at `input * 2^i`, not `input / 2^i`. Getting this backwards
/// is a size error at the second level rather than a subtle one, which is how it
/// was found.
fn draw_eps(w: &weights::Weights, in_h: usize, in_w: usize, seed: u64) -> Vec<net::Plane> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(w.config.n_levels);
    for level in 0..w.config.n_levels {
        let h = in_h << level;
        let ww = in_w << level;
        let c = w.config.cond_in[level];
        let mut p = net::Plane::new(c, h, ww);
        for v in p.data.iter_mut() {
            *v = rng.next_normal();
        }
        out.push(p);
    }
    out
}

/// Read a `.npy` f32 array of shape (1, 3, h, w) - the numpy subset needed to
/// cross-check against the torch reference. Only the one dtype and the C-order
/// case are supported, and it fails loudly otherwise: this is a validation path,
/// not a general reader.
fn read_npy(path: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{path}: not a .npy file"));
    }
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen])
        .map_err(|e| format!("{path}: header: {e}"))?;
    if !header.contains("'<f4'") {
        return Err(format!("{path}: only '<f4' arrays are supported ({header})"));
    }
    if header.contains("'fortran_order': True") {
        return Err(format!("{path}: Fortran order is not supported"));
    }
    let open = header.find('(').ok_or_else(|| format!("{path}: no shape"))?;
    let close = header.find(')').ok_or_else(|| format!("{path}: no shape"))?;
    let shape: Vec<usize> = header[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    let data = &bytes[hstart + hlen..];
    let n: usize = shape.iter().product();
    if data.len() != n * 4 {
        return Err(format!("{path}: {} payload bytes for {n} elements", data.len()));
    }
    let mut v = Vec::with_capacity(n);
    for c in data.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((shape, v))
}

/// Estimated peak memory for one forward pass, in bytes, from the LR input size.
///
/// CALIBRATED, not derived. The graph's working set is not a clean multiple of
/// the input: every level's planes scale with `lr^2` but the plan sizes are
/// bucket-quantised, so a purely analytic sum would be wrong in the direction
/// that matters. What this is instead is the bounding envelope of MEASURED
/// peaks at thirteen sizes from 64x64 to 512x512 (peak live device bytes from
/// the pool's own high-water mark, and peak host RSS from `/usr/bin/time -v`),
/// fitted as `S * lr^2 + C`:
///
///     lr    64    96   112   128   144   160   192   224   256   320   384   448   512
///     gpu  140   202   248   286   346   400   533   694   875  1318  1641  2202  2836 MiB
///     cpu  154   244   301   367   441   474   609   754  1027  1533  2132  2826  3788 MiB
///
/// A line through the two LARGEST points is the asymptotic slope; the envelope
/// is the least slope that still lies above every measurement, which is why the
/// estimate is 25-31% generous at 512 and tight (1.0-1.1x) at 64-256. Being
/// generous is the right side to err on for a guard: a false "not enough memory"
/// is a clear message, whereas an under-estimate is the crash this exists to
/// prevent.
fn estimated_peak_bytes(lr_pixels: usize, gpu: bool, weights_bytes: u64) -> u64 {
    const MIB: f64 = 1024.0 * 1024.0;
    let (slope, base) = if gpu {
        // Per LR pixel, plus the fixed floor (kernels, pool bookkeeping).
        (0.013822_f64, 83.4_f64)
    } else {
        (0.017773_f64, 81.2_f64)
    };
    let pooled = slope * lr_pixels as f64 + base;
    let mut bytes = (pooled * MIB) as u64;
    // The pooled peak counts POOL blocks only. The checkpoint's own device
    // residency is a separate allocation (88.8 MiB for the x4 model), and on
    // the device it is resident for the whole run.
    if gpu {
        bytes += weights_bytes;
    }
    bytes
}

/// Host memory that can be handed out without pushing the machine into swap.
///
/// `MemAvailable` is the kernel's own estimate and already accounts for page
/// cache that can be reclaimed, so it is the right number rather than `MemFree`.
/// If the caller has a finite address-space limit it is a hard ceiling too, and
/// the lower of the two is what matters - a job under `ulimit -v` must fail on
/// the limit rather than on the swap it was never allowed to reach.
fn available_host_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let avail = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())?
        .saturating_mul(1024);
    // A finite RLIMIT_AS is a ceiling on the whole process's address space; a
    // run can only be as large as the limit minus what is already mapped.
    let limit = unsafe {
        let mut rl: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_AS, &mut rl) == 0 && rl.rlim_cur != libc::RLIM_INFINITY {
            Some(rl.rlim_cur as u64)
        } else {
            None
        }
    };
    Some(match limit {
        Some(l) => avail.min(l),
        None => avail,
    })
}

/// Refuse a run that provably cannot fit, BEFORE it allocates anything.
///
/// Without this the failure lands in the middle of the graph: on the device as a
/// `cuMemAlloc` error several hundred allocations in, and on the host as an
/// abort from the allocator (the profile is `panic = "abort"`, so a failed `Vec`
/// is not even catchable). Neither is a clear message and the host one can leave
/// the machine paging before it dies, which is the thrashing this avoids.
fn memory_guard(lr_h: usize, lr_w: usize, gpu: bool, weights_bytes: u64) -> Result<(), String> {
    let need = estimated_peak_bytes(lr_h * lr_w, gpu, weights_bytes);
    let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    let backend = if gpu { "gpu" } else { "cpu" };
    let avail: u64 = if gpu {
        match lightgpu::vm::free_vram() {
            Ok(a) => a as u64,
            // No driver, no context, or a machine without a GPU: the CPU
            // backend's own guard applies instead, and there is nothing to
            // check here.
            Err(_) => return Ok(()),
        }
    } else {
        match available_host_bytes() {
            Some(a) => a,
            // No /proc/meminfo (not Linux): nothing to check against.
            None => return Ok(()),
        }
    };
    if need <= avail {
        return Ok(());
    }
    // GiB reads better above a gigabyte and MiB below it, and the number is the
    // point of the message, so it is formatted in whatever unit it lands in.
    let size = |b: u64| {
        if b >= 1024 * 1024 * 1024 {
            format!("{:.2} GiB", gb(b))
        } else {
            format!("{:.0} MiB", mib(b))
        }
    };
    Err(format!(
        "not enough memory for a {lr_h}x{lr_w} input on the {backend} backend\n  \
         this input needs about {}, from the measured peak at every size from 64x64 to 512x512\n  \
         only {} is {}\n  \
         use a smaller input, or free {} memory, or run on the {} backend",
        size(need),
        size(avail),
        if gpu { "free on the device" } else { "available on this machine" },
        if gpu { "device" } else { "machine" },
        if gpu { "cpu" } else { "gpu (--device gpu)" }
    ))
}

/// `--ref-run <lr.npy> <ref.npy>`: run the graph on the LR tensor a torch
/// reference consumed at eps_std 0 and report the difference against its
/// output. This is the end-to-end numerical check of the port: the reference
/// numbers are the framework's own forward pass, not a reimplementation.
fn ref_run(w: &weights::Weights, lr_path: &str, ref_path: &str) {
    let (shape, data) = match read_npy(lr_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        eprintln!("error: {lr_path} has shape {shape:?}, expected (1, 3, h, w)");
        std::process::exit(1);
    }
    let (h, wd) = (shape[2], shape[3]);
    let lr = net::Plane::from_vec(3, h, wd, data);
    let opts = net::Options::with_eps(0.0, Vec::new());
    let out = match net::forward_cpu(w, &lr, &opts, |_| {}) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    let (rshape, rdata) = match read_npy(ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if rshape.len() != 4 || rshape != vec![1, 3, h * 4, wd * 4] {
        eprintln!("error: reference shape {rshape:?}, engine gave (1, 3, {}, {})", h * 4, wd * 4);
        std::process::exit(1)
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut worst = 0usize;
    for (i, (a, b)) in out.data.iter().zip(rdata.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        sum_abs += d as f64;
    }
    let mean = sum_abs / out.data.len() as f64;
    println!(
        "reference check: {}x{}x{} vs {}x{}x{}, max|diff| {:.3e} (at element {worst}), mean|diff| {:.3e}",
        out.c,
        out.h,
        out.w,
        rshape[1],
        rshape[2],
        rshape[3],
        max_abs,
        mean
    );
    if max_abs > 2e-3 {
        eprintln!("FAIL: the engine disagrees with the reference by more than 2e-3");
        std::process::exit(1);
    }
    println!("reference check ok");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut model_path = None;
    let mut input = None;
    let mut output = None;
    // A CPU-only build has no GPU backend to default to, so it defaults to the
    // one it can actually run.
    let mut device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" }.to_string();
    // Set only when the caller NAMED the GPU. Without it, a GPU that cannot be
    // brought up is not fatal: the engine falls back to the CPU backend, which
    // is what lets one binary run on a machine with no NVIDIA driver at all.
    let mut force_gpu = false;
    let mut eps_std: Option<f32> = None;
    let mut seed: Option<u64> = None;
    let mut quiet = false;
    let mut cpu_selftest = false;
    let mut trace = false;
    let mut prim = false;
    let mut ng_test = false;
    let mut conv_bench: Option<[usize; 5]> = None;
    let mut selftest = false;
    let mut parity = 0usize;
    let mut ref_run_args: Option<(String, String)> = None;
    // `--to-host` also DOWNLOADS the output plane to the host. `forward in Xs`
    // otherwise times the graph plus one device synchronise, with the image
    // marshalling and the PNG write outside the measurement - which is the
    // right thing for comparing kernels and the wrong thing for comparing
    // against a reference whose number includes its own host transfer.
    let mut to_host = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            "-i" | "--input" => {
                i += 1;
                input = args.get(i).cloned();
            }
            "-o" | "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            "--device" => {
                i += 1;
                device = args.get(i).cloned().unwrap_or_else(|| usage());
                force_gpu = device == "gpu";
            }
            "--cpu" => device = "cpu".to_string(),
            "--gpu" => {
                device = "gpu".to_string();
                force_gpu = true;
            }
            "--eps-std" | "--eps_std" => {
                i += 1;
                eps_std = args.get(i).and_then(|s| s.parse().ok());
                if eps_std.is_none() {
                    usage();
                }
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok());
                if seed.is_none() {
                    usage();
                }
            }
            "-q" | "--quiet" => quiet = true,
            "--self-test" => cpu_selftest = true,
            // Include the device-to-host download and the host clamp in the
            // reported forward time. The default excludes them, which is right
            // for comparing kernels and wrong for comparing against a reference
            // whose own number includes its host transfer.
            "--to-host" => to_host = true,
            "--trace" => trace = true,
            "--prim-test" => prim = true,
            "--ng-test" => ng_test = true,
            "--conv-bench" => {
                // `--conv-bench` alone benches the shapes the graph runs; with
                // five numbers it benches exactly `c_in c_out h w iters`.
                let nums: Vec<usize> = args[i + 1..]
                    .iter()
                    .take_while(|s| s.chars().all(|c| c.is_ascii_digit()))
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += nums.len();
                conv_bench = Some(if nums.len() >= 4 {
                    let g = |n: usize, d: usize| nums.get(n).copied().unwrap_or(d);
                    [nums[0], nums[1], nums[2], nums[3], g(4, 20)]
                } else {
                    [0, 0, 0, 0, 20]
                });
            }
            "--ref-run" => {
                let a = args.get(i + 1).cloned().unwrap_or_else(|| usage());
                let b = args.get(i + 2).cloned().unwrap_or_else(|| usage());
                ref_run_args = Some((a, b));
                i += 2;
            }
            "--cuda-selftest" => selftest = true,
            "--parity" => {
                i += 1;
                parity = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "-h" | "--help" => usage(),
            "-V" | "--version" => {
                println!("hcflow {VERSION}");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                usage();
            }
        }
        i += 1;
    }
    if !cfg!(feature = "cuda") {
        // A build without the feature has only the CPU path, and the usage text
        // promises `--gpu` "refuses to fall back": silently ignoring the request
        // would run the CPU backend while the command line said gpu. The default
        // device is already `cpu` in this build, so this only fires on an
        // explicit `--gpu`/`--device gpu`.
        if force_gpu {
            eprintln!("error: this build has no `cuda` feature; use --device cpu");
            std::process::exit(2);
        }
    }

    if let Some(b) = conv_bench {
        #[cfg(feature = "cuda")]
        {
            let ca = match cuda::Cuda::init(quiet) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1)
                }
            };
            let shapes: Vec<[usize; 4]> = if b[0] > 0 {
                vec![[b[0], b[1], b[2], b[3]]]
            } else {
                // The graph's own shapes at lr64: the level-0 trunk (64 channels
                // at 64x64), the level-1 trunk (64 at 128x128), the two feature
                // convs (3 -> 64 and 128 -> 64) and the conditional-out conv
                // (128 -> 42).
                vec![
                    [3, 64, 64, 64],
                    [64, 64, 64, 64],
                    [128, 64, 64, 64],
                    [64, 64, 128, 128],
                    [128, 42, 64, 64],
                ]
            };
            for s in shapes {
                if let Err(e) = gpu::conv_bench(&ca, s[0], s[1], s[2], s[3], b[4]) {
                    eprintln!("error: {e}");
                    std::process::exit(1)
                }
            }
            return;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = b;
            eprintln!("error: --conv-bench needs the `cuda` feature");
            std::process::exit(1);
        }
    }
    if ng_test {
        #[cfg(feature = "cuda")]
        {
            let ca = match cuda::Cuda::init(!quiet) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1)
                }
            };
            if let Err(e) = gpu::ng_selftest(&ca, !quiet) {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
            return;
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("error: --ng-test needs the `cuda` feature");
            std::process::exit(1);
        }
    }
    // The model is loaded HERE rather than before the two branches above, because
    // --conv-bench and --ng-test take no checkpoint: the bench uses synthetic
    // data and its own weights, and the ng test compares two kernels on fixed
    // data. Requiring `-m` for them made the usage text ("no model needed") a
    // promise the parser did not keep.
    let model_path = model_path.unwrap_or_else(|| usage());
    let w = match weights::Weights::open(&model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if !quiet {
        eprintln!(
            "model: {} ({:.1} MiB, {} tensors, scale x{}, {} levels, {} RRDB blocks/trunk, eps_std {})",
            model_path,
            w.total_bytes() as f64 / (1024.0 * 1024.0),
            w.num_tensors(),
            w.config.scale,
            w.config.n_levels,
            w.config.rrdb_blocks,
            w.config.eps_std
        );
        eprintln!(
            "graph: in_ch {:?}, c_split {:?}, cond_in {:?}, cond_out {:?}",
            w.config.in_ch, w.config.c_split, w.config.cond_in, w.config.cond_out
        );
    }

    if prim {
        #[cfg(feature = "cuda")]
        {
            let ca = match cuda::Cuda::init(!quiet) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1)
                }
            };
            if let Err(e) = gpu::prim_selftest(&ca, !quiet) {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
            return;
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("error: --prim-test needs the `cuda` feature");
            std::process::exit(1);
        }
    }
    if trace {
        let n = std::env::var("HCFLOW_TRACE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        trace_run(&w, n);
        return;
    }
    if cpu_selftest {
        cpu_selftest_run(&w);
        return;
    }
    if let Some((lr, rf)) = &ref_run_args {
        ref_run(&w, lr, rf);
        return;
    }

    if selftest || parity > 0 {
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("error: --cuda-selftest and --parity need a build with the `cuda` feature");
            std::process::exit(1);
        }
        #[cfg(feature = "cuda")]
        {
            let n = if parity > 0 { parity } else { 32 };
            let eps = eps_std.unwrap_or(w.config.eps_std);
            if let Err(e) = parity_run(&w, n, eps, quiet) {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
            return;
        }
    }

    let eps_std = eps_std.unwrap_or(w.config.eps_std);
    // `-` and an unspecified path both mean the standard stream, so the
    // program composes in a pipe: `... -o - | ...`
    let input = input.unwrap_or_else(|| "-".to_string());
    let output = output.unwrap_or_else(|| "-".to_string());
    let img = match if input == "-" {
        image::load_rgb_stream(std::io::stdin().lock()).map_err(|e| format!("stdin: {e}"))
    } else {
        image::load_rgb(&input)
    } {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    if !quiet {
        let src = if input == "-" { "stdin".to_string() } else { input.clone() };
        eprintln!("input: {} ({}x{})", src, img.w, img.h);
    }
    if img.h % 4 != 0 || img.w % 4 != 0 {
        eprintln!(
            "error: the x4 model needs an input that is a multiple of 4 in each axis, got {}x{}",
            img.w, img.h
        );
        std::process::exit(1);
    }

    // Both backends draw from the SAME seeded generator, so a run with a seed
    // is reproducible in either one - which is also what makes `--parity` a
    // comparison of the graph rather than of two random draws.
    let opts = match seed {
        Some(s) => net::Options::with_eps(eps_std, draw_eps(&w, img.h, img.w, s)),
        None => net::Options::sampled(eps_std),
    };
    if !quiet && eps_std == 0.0 {
        eprintln!("eps_std 0: the map is deterministic (the reference's mean mode)");
    }

    HOST_SECS.with(|c| c.set(0.0));
    let t_fwd = Instant::now();
    let out = match device.as_str() {
        "cpu" => run_cpu(&w, &img, &opts, quiet),
        "gpu" => {
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("error: this binary was built without the `cuda` feature; use --device cpu");
                std::process::exit(1);
            }
            #[cfg(feature = "cuda")]
            {
                match run_gpu(&w, &img, &opts, quiet) {
                    Ok(v) => v,
                    // NAMING THE GPU IS A REQUEST; NOT NAMING IT IS NOT. gpu is
                    // the default, so a machine with no driver must still work:
                    // the driver failing to come up is not an error unless the
                    // caller asked for the GPU by name.
                    //
                    // AN OUT-OF-MEMORY ERROR IS THE EXCEPTION, even unnamed. The
                    // fallback exists for a machine with no usable driver, whose
                    // failure happens before any work; running out of memory is
                    // not that, and quietly moving a job the CPU cannot hold onto
                    // the CPU is exactly the thrash the memory guard refuses.
                    Err(e) if force_gpu || e.contains("not enough memory") || e.contains("CUDA_ERROR_OUT_OF_MEMORY") => {
                        eprintln!("error: {e}");
                        std::process::exit(1)
                    }
                    Err(e) => {
                        eprintln!("cuda: {e}");
                        eprintln!("cuda: falling back to the CPU backend (--gpu forces the GPU)");
                        run_cpu(&w, &img, &opts, quiet)
                    }
                }
            }
        }
        other => {
            eprintln!("unknown device: {other}");
            usage()
        }
    };
    if !quiet {
        eprintln!("forward in {:.2}s", t_fwd.elapsed().as_secs_f64());
        if to_host {
            let hs = HOST_SECS.with(|c| c.get());
            if to_host {
                eprintln!("  of which device->host download: {:.2}s", hs);
            }
        }
    }
    // The allocator's high-water mark is the number the memory budget is
    // stated in, and it is only observable from here: `peak` counts what the
    // pool has ever had resident, which is the true footprint of a pass.
    // The device-pool report and the leak trace are the CUDA backend's, so a
    // build without the feature (CPU only) has nothing to say here.
    #[cfg(feature = "cuda")]
    {
        if !quiet {
            let (hits, misses, live, free, peak) = cuda::pool_stats();
            // `live` is normally 0 here - every plane has been dropped by now -
            // so the numbers that mean anything are the peak of what was live at
            // once (the graph's working set) and the free list (which is also
            // device memory, held by the pool rather than by the graph).
            eprintln!(
                "device pool: {hits} hits, {misses} misses, peak {} MiB live, {} MiB \
                 held by the pool ({} MiB live now)",
                peak / (1024 * 1024),
                free / (1024 * 1024),
                live / (1024 * 1024)
            );
        }
        cuda::leak_report();
    }
    let result = image::Image { w: out.w, h: out.h, data: out.data };
    let rgb = result.to_rgb8();
    if let Err(e) = if output == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        image::save_rgb_stream(&mut lock, result.w, result.h, &rgb)
            .and_then(|()| std::io::Write::flush(&mut lock).map_err(|e| e.to_string()))
            .map_err(|e| format!("stdout: {e}"))
    } else {
        image::save_rgb(&output, result.w, result.h, &rgb)
    } {
        eprintln!("error: {e}");
        std::process::exit(1)
    }
    if !quiet {
        let dst = if output == "-" { "stdout".to_string() } else { output.clone() };
        eprintln!("wrote {} ({}x{})", dst, result.w, result.h);
    }
}

/// The CPU forward pass.
///
/// One function rather than a block inside the match, because it runs on two
/// paths that must agree: `--device cpu`, and the fallback from a GPU that could
/// not be brought up. A second copy would be a second set of progress output to
/// keep in step.
fn run_cpu(w: &weights::Weights, img: &image::Image, opts: &net::Options, quiet: bool) -> net::Plane {
    // The host side aborts rather than erroring when `Vec` cannot allocate (the
    // release profile is `panic = "abort"`), and it would have to PACE THE
    // MACHINE for a while first - which is the thrashing this refuses to do.
    // This runs on both CPU paths: `--device cpu` and the fallback from a GPU
    // that could not hold the run.
    if let Err(e) = memory_guard(img.h, img.w, false, 0) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    let p = net::Plane { c: 3, h: img.h, w: img.w, data: img.data.clone() };
    let mut n = 0;
    let mut progress = |s: &str| {
        if !quiet {
            n += 1;
            eprintln!("  [{n}] {s}");
        }
    };
    match net::forward_cpu(w, &p, opts, &mut progress) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    }
}

#[cfg(feature = "cuda")]
fn run_gpu(w: &weights::Weights, img: &image::Image, opts: &net::Options, quiet: bool) -> Result<net::Plane, String> {
    // BEFORE the module is loaded or a byte of device memory is asked for. The
    // checkpoint's device residency is a separate allocation from the pool's
    // working set, so it is added on top rather than assumed inside.
    memory_guard(img.h, img.w, true, w.total_bytes() as u64)?;
    let ca = cuda::Cuda::init(!quiet)?;
    ca.upload_weights(&w.file)?;
    let lr = gpu::Plane::from_host(&img.data, 3, img.h, img.w)?;
    let mut n = 0;
    let mut progress = |s: &str| {
        if !quiet {
            n += 1;
            eprintln!("  [{n}] {s}");
        }
    };
    let y = gpu::forward_gpu(&ca, w, &lr, opts, &mut progress)?;
    ca.sync()?;
    ca.profile_report();
    let t0 = Instant::now();
    let host = y.to_host()?;
    HOST_SECS.with(|c| c.set(c.get() + t0.elapsed().as_secs_f64()));
    Ok(net::Plane { c: y.c, h: y.h, w: y.w, data: host })
}

/// Run both backends on one seeded eps draw and report the difference.
///
/// This is the engine's numerical evidence: the two backends implement the same
/// algebra through different code (host loops against CUDA kernels), so a
/// disagreement at a few 1e-3 in fp32 means one of them is walking the graph
/// differently - and a disagreement that GROWS with the depth of the walk means
/// an accumulation-order difference that the tolerance below is about.
#[cfg(feature = "cuda")]
fn parity_run(w: &weights::Weights, n: usize, eps_std: f32, quiet: bool) -> Result<(), String> {
    // The floor is 16, not 8: the test image below is band-limited in
    // NORMALISED coordinates, so at 8x8 its 5-cycle term is above Nyquist and
    // aliases into the per-pixel noise that the comment on that pattern exists
    // to avoid - `--parity 8` would then fail for a reason that has nothing to
    // do with the two backends disagreeing.
    let nn = n.max(16) & !3;
    // A BAND-LIMITED image, not per-pixel noise. The conditional flow samples
    // `mean + exp(eps_std * logs) * eps` and the network was trained on
    // downsampled natural images, so a per-pixel pseudo-random pattern drives
    // the latent far outside its trained range: at eps_std 0.9 the REFERENCE
    // MODEL ITSELF overflows fp32 on such an input and produces NaN. Sums of a
    // few low-frequency sinusoids then quantised to 8 bits stay in range at both
    // temperatures, which is what makes this a comparison of two backends rather
    // than of two rounding paths through a chaotic blow-up.
    let mut img = image::Image::new(nn, nn);
    let hw = nn * nn;
    let band = |x: usize, y: usize, k: f32, phase: f32| -> f32 {
        let t = (x as f32) / nn as f32;
        let s = (y as f32) / nn as f32;
        let v = (k * std::f32::consts::TAU * t + phase).sin()
            * (0.6 * k * std::f32::consts::TAU * s + 0.5 * phase).cos();
        // Quantise to 8 bits the way a real PNG would, so the comparison is not
        // between two float inputs no file could hold.
        ((v * 0.5 + 0.5) * 255.0).round() / 255.0
    };
    for y in 0..nn {
        for x in 0..nn {
            img.data[y * nn + x] = band(x, y, 2.0, 0.0);
            img.data[hw + y * nn + x] = band(x, y, 3.0, 1.1);
            img.data[2 * hw + y * nn + x] = band(x, y, 5.0, 2.3);
        }
    }
    let opts = net::Options::with_eps(eps_std, draw_eps(w, nn, nn, 12345));
    let t0 = Instant::now();
    let cpu = net::forward_cpu(
        w,
        &net::Plane { c: 3, h: nn, w: nn, data: img.data.clone() },
        &opts,
        |_| {},
    )?;
    let cpu_ms = t0.elapsed().as_secs_f64();

    let ca = cuda::Cuda::init(!quiet)?;
    ca.upload_weights(&w.file)?;
    let lr = gpu::Plane::from_host(&img.data, 3, nn, nn)?;
    let mut gpu = {
        let mut marks = Vec::new();
        let y = gpu::forward_gpu(&ca, w, &lr, &opts, |s| marks.push(s.to_string()))?;
        let host = net::Plane { c: y.c, h: y.h, w: y.w, data: y.to_host()? };
        (host, marks)
    };
    let gpu_ms = t0.elapsed().as_secs_f64() - cpu_ms;

    let (gpu_out, marks) = (&mut gpu.0, &mut gpu.1);
    // A non-finite output would otherwise pass: every comparison against NaN is
    // false, so `max` would stay 0 and the mean would be NaN - a run that
    // exploded would report "parity ok". Both backends are checked here rather
    // than only the difference.
    let cpu_nf = cpu.data.iter().filter(|v| !v.is_finite()).count();
    let gpu_nf = gpu_out.data.iter().filter(|v| !v.is_finite()).count();
    println!("  non-finite: cpu {cpu_nf}/{} gpu {gpu_nf}/{}", cpu.data.len(), gpu_out.data.len());
    if cpu_nf > 0 || gpu_nf > 0 {
        return Err(format!("non-finite output (cpu {cpu_nf}, gpu {gpu_nf}) - parity is meaningless"));
    }
    let mut max = 0f32;
    let mut at = 0usize;
    for (i, (a, b)) in cpu.data.iter().zip(gpu_out.data.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max {
            max = d;
            at = i;
        }
    }
    let mean = cpu.data.iter().zip(gpu_out.data.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>()
        / cpu.data.len() as f32;
    println!("parity on {}x{} (eps_std {})", nn, nn, eps_std);
    println!("  stages: {}", marks.join(" -> "));
    println!("  cpu forward {:.3}s", cpu_ms);
    println!("  gpu forward {:.3}s (including its weight upload)", gpu_ms);
    println!("  max |cpu - gpu| = {max:.3e} at element {at}, mean {mean:.3e}");
    println!(
        "  cpu range [{:.4}, {:.4}]",
        cpu.data.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    if !max.is_finite() {
        return Err("the comparison produced a non-finite difference".to_string());
    }
    // fp32 through 26 flow steps and 420 convolutions: a few 1e-3 is the
    // expected size of a difference between two different accumulation orders,
    // and 41x on the input size has produced no worse.
    if max > 5e-2 {
        return Err(format!("backends disagree by {max:.3e}"));
    }
    println!("  parity ok");
    ca.profile_report();
    Ok(())
}

/// Write a plane as a `.npy` f32 array of shape (1, c, h, w), for `--trace` with
/// HCFLOW_TRACE_DUMP: the only way to compare a stage with the framework's own
/// intermediate element by element rather than by range.
fn save_npy(path: &str, p: &net::Plane) -> std::io::Result<()> {
    use std::io::Write;
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': (1, {}, {}, {}), }}", p.c, p.h, p.w);
    let pad = 64 - ((10 + header.len() + 1) % 64);
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;
    let total = header.len() + pad + 1;
    f.write_all(&(total as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(&vec![b' '; pad])?;
    f.write_all(b"\n")?;
    let mut bytes = Vec::with_capacity(p.data.len() * 4);
    for v in &p.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&bytes)
}

/// Report a plane's size and the range of its values, for `--trace`.
fn stat(name: &str, p: &net::Plane) {
    let mx = p.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mn = p.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let nan = p.data.iter().filter(|v| !v.is_finite()).count();
    eprintln!(
        "  {name:<28} {}x{}x{}  [{mn:+.4}, {mx:+.4}]{}",
        p.c,
        p.h,
        p.w,
        if nan > 0 { format!("  {nan} NON-FINITE") } else { String::new() }
    );
}

/// Walk one level of the graph stage by stage and print each stage's statistics.
///
/// The forward pass is a chain of twenty-odd stages per level, and a non-finite
/// value only says WHERE it appeared, not WHY: the first stage that blows up is
/// what identifies a wrong direction (an inverted ActNorm or 1x1 decays for a
/// while and then explodes) as opposed to a wrong shape or a wrong concat order.
fn trace_run(w: &weights::Weights, n: usize) {
    let dump_dir = std::env::var("HCFLOW_TRACE_DUMP").ok();
    macro_rules! dump {
        ($name:expr, $plane:expr) => {
            if let Some(d) = &dump_dir {
                let p = format!("{d}/{}.npy", $name);
                if let Err(e) = save_npy(&p, $plane) {
                    eprintln!("dump {p}: {e}");
                }
            }
        };
    }
    let cfg = &w.config;
    // HCFLOW_TRACE_NPY points at a reference LR tensor, so a trace can be
    // compared stage by stage with the framework's own forward pass.
    let lr = match std::env::var("HCFLOW_TRACE_NPY") {
        Ok(p) => {
            let (shape, data) = read_npy(&p).expect("read HCFLOW_TRACE_NPY");
            assert_eq!(shape, vec![1, 3, n, n], "trace npy shape");
            net::Plane::from_vec(3, n, n, data)
        }
        Err(_) => net::Plane { c: 3, h: n, w: n, data: vec![0.5f32; 3 * n * n] },
    };
    let lr = lr;
    eprintln!(
        "  input range [{:+.4}, {:+.4}]",
        lr.data.iter().cloned().fold(f32::INFINITY, f32::min),
        lr.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    // A trace driven from a reference tensor is compared against the reference's
    // eps_std = 0 run, so it must use the same temperature; the free trace uses
    // the checkpoint's own eps_std with a seeded draw.
    let eps_std = match std::env::var("HCFLOW_TRACE_EPS") {
        Ok(v) => v.parse().unwrap_or(cfg.eps_std),
        Err(_) => match std::env::var("HCFLOW_TRACE_NPY") {
            Ok(_) => 0.0,
            Err(_) => cfg.eps_std,
        },
    };
    let (eps_std, eps) = (eps_std, draw_eps(w, n, n, 3));
    let opts = net::Options::with_eps(eps_std, eps);
    stat("input", &lr);
    let mut z = lr;
    let mut prev: Option<net::Plane> = None;
    for level in 0..cfg.n_levels {
        eprintln!("level {level} ({} channels in):", z.c);
        let u = match (&prev, cfg.cond_levels[level]) {
            (Some(f), k) if k > 0 => net::cat(&z, &net::upsample2x(f)),
            _ => z.clone(),
        };
        stat("  conditioning u", &u);
        dump!(format!("l{level}_u"), &u);

        let (f1, f2) = match net::cond_feature_sr(w, level, &u) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return;
            }
        };
        stat("  feature trunk0", &f1);
        stat("  feature trunk_conv1+res", &f2);
        let feature = net::cat(&f1, &f2);
        stat("  feature (128ch)", &feature);
        dump!(format!("l{level}_feature"), &feature);

        let ci = cfg.cond_in[level];
        let b = format!("l{level}");
        let ow = w.get(&format!("{b}.cond_out_w")).unwrap();
        let ob = w.get(&format!("{b}.cond_out_b")).unwrap();
        let h = net::conv3x3(&feature, ow, Some(ob), 2 * ci);
        stat("  f(feature)", &h);

        let eps = opts.eps.as_ref().and_then(|v| v.get(level));
        let a = match net::cond_flow(w, level, &z, &u, &opts, eps) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return;
            }
        };
        stat("  sampled + cond steps", &a);
        dump!(format!("l{level}_a"), &a);
        z = net::cat(&z, &a);
        stat("  after Split inverse", &z);
        dump!(format!("l{level}_split"), &z);
        let full = cfg.full_ch[level];
        for i in (0..cfg.steps[level]).rev() {
            let s = match net::load_step_n(w, &weights::step(&b, i), full) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {e}");
                    return;
                }
            };
            z = net::flow_step_rev(&s, &z, None);
            if i == cfg.steps[level] - 1 || i == 0 {
                dump!(format!("l{level}_step{i}"), &z);
                stat(&format!("  step {i} reversed"), &z);
            }
        }
        z = net::unsqueeze2d(&z);
        stat("  after unsqueeze2d", &z);
        dump!(format!("l{level}_unsqueeze"), &z);
        // The NEXT level conditions on this level's 128-channel FEATURE (the
        // reference's `conditional_feature2`), not on the conditioning input
        // `u` the feature was built from: passing `u` on would silently give
        // the inner level 3 channels where the graph expects 128.
        prev = Some(feature);
    }
}

/// The CPU backend's own check: the graph runs, the output has the right shape,
/// is finite, and is not constant.
fn cpu_selftest_run(w: &weights::Weights) {
    let cfg = &w.config;
    println!(
        "cpu selftest: scale x{}, {} levels, eps_std {}",
        cfg.scale, cfg.n_levels, cfg.eps_std
    );
    let n = 16usize;
    let img = net::Plane { c: 3, h: n, w: n, data: vec![0.5f32; 3 * n * n] };
    let opts = net::Options::with_eps(cfg.eps_std, draw_eps(w, n, n, 7));
    let mut stages = Vec::new();
    let mut progress = |s: &str| stages.push(s.to_string());
    let out = match net::forward_cpu(w, &img, &opts, &mut progress) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1)
        }
    };
    println!("stages: {}", stages.join(" -> "));
    println!("output: {}x{}x{}", out.c, out.h, out.w);
    assert_eq!(out.c, 3, "output channels");
    assert_eq!(out.h, n * cfg.scale, "output height");
    assert_eq!(out.w, n * cfg.scale, "output width");
    assert!(out.data.iter().all(|v| v.is_finite()), "output has a non-finite value");
    assert!(out.data.iter().all(|v| *v >= 0.0 && *v <= 1.0), "output left [0, 1]");
    let mn = out.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("range [{mn:.4}, {mx:.4}]");
    // A constant 0.5 input with a seeded eps must not give a constant output:
    // the network is not shift-invariant-free, so a constant result means a
    // stage was skipped.
    assert!(mx - mn > 1e-6, "output is constant: spread {}", mx - mn);
    println!("cpu selftest ok");
}
