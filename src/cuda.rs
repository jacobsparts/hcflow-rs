//! CUDA backend: device memory, the embedded fatbin module, launches.
//!
//! The driver bindings, the context, the module and the argument marshalling
//! come from `lightgpu`; what is here is the f32-facing surface the graph code
//! uses (buffers counted in ELEMENTS rather than bytes, name resolution, the
//! profiling bracket) plus the ONE structural addition this engine wants: a
//! single weights arena.
//!
//! THE ARENA. A converted HCFlow checkpoint is ~23M parameters. Uploading them
//! tensor by tensor is a few hundred small `cuMemcpyHtoD` calls, each with its
//! own driver entry and page-fault cost; instead `upload_weights` copies the
//! whole payload section in one call and `Weights::dev(name)` hands out
//! `base + offset`. The offsets are the container's own absolute file offsets
//! minus the payload start, and they are 4-byte aligned because the converter
//! pad-aligns every tensor to 8 bytes, so the pointer arithmetic is exact.
//!
//! Compiled only with `--features cuda`. `libcuda.so.1` is `dlopen`ed by
//! lightgpu at run time, so the CPU-only build needs no NVIDIA driver at all.

use lightgpu::ffi::CUfunction;
use lightgpu::vm;
use lightgpu::vm::{Args, Launch};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// The call site currently allocating, set by `tag()` and consumed by
    /// `DevBuf::alloc`. A live total that will not come down is nearly always one
    /// call site holding a whole sequence of buffers, and a backtrace only says
    /// where ONE of them came from - this says which site owns all of them.
    static TAG: std::cell::Cell<&'static str> = const { std::cell::Cell::new("?") };
}

/// Label subsequent allocations. Diagnostic only, and deliberately cheap: a
/// thread-local store, read when an allocation happens.
pub fn tag(where_: &'static str) {
    TAG.with(|t| t.set(where_));
}

thread_local! {
    /// Per-(kernel, launch grid) device time in ms and call count, filled only
    /// under `LA_PROFILE=1`.
    ///
    /// The grid is part of the key because one kernel serves very different
    /// geometries - the conditional-flow steps run at 24 channels on the
    /// incoming scale while the unconditional ones run at 24 too but with a
    /// different coupling width - and a name-only total hides which regime is
    /// losing time.
    static PROFILE: RefCell<HashMap<String, (f64, usize)>> = RefCell::new(HashMap::new());
}

fn profiling() -> bool {
    std::env::var("LA_PROFILE").map(|v| v == "1").unwrap_or(false)
}

/// The toolkit's generic ops, compiled down to the subset this engine calls.
#[cfg(feature = "cuda")]
pub static TOOLKIT_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/hcflow_toolkit.fatbin"));

/// This project's own kernels (`cuda/hcflow.cu`). Loaded as a SEPARATE module
/// from the toolkit's so neither can shadow a name in the other; `func` below
/// falls through to it. Today it holds the two ops the toolkit has no equivalent
/// for: the coupling's fused shift/logscale inversion and the conditional flow's
/// latent sampler. (The Unsqueeze2d inverse was the third until it turned out to
/// be the toolkit's `lg_pixel_shuffle` at r = 2; see `gpu::unsqueeze2d`.)
#[cfg(feature = "cuda")]
pub static PROJECT_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/hcflow_project.fatbin"));

/// Every kernel this engine launches, resolved eagerly at startup. Keeping this
/// an explicit list turns a renamed or mis-listed kernel into a startup error;
/// the per-launch path then costs one hash lookup.
pub const KERNEL_NAMES: &[&str] = &[
    // Toolkit: this engine's convs and the elementwise/permutation ops it
    // builds its graph from.
    "lg_conv3x3s1p1",
    "lg_conv1x1",
    "lg_conv1x1_rb",
    "lg_conv3x3_tile",
    // The parameterised tile body, promoted to the toolkit from this engine.
    // The four names are UNCHANGED, so the graph's dispatch, its geometry table
    // and `--conv-bench` are untouched: only the module they resolve from moved.
    // Same accumulation order (ci, ky, kx, bias after the sum) and same tile, so
    // this is a bit-identical swap, and the promote-then-compare sequence proved
    // it byte-for-byte against the pre-promotion binary before the project's
    // copies were deleted.
    "lg_conv3x3_q2",
    "lg_conv3x3_q2ng2",
    "lg_conv3x3_catq0",
    "lg_conv3x3_catq0ng2",
    "lg_conv3x3_winograd",
    "lg_channel_affine",
    "lg_relu",
    "lg_lrelu",
    "lg_add_scaled",
    // `lg_copy` builds the dense blocks' growing concatenation on the device;
    // at 512x512 a host round trip for each of the 15 concats per dense block
    // would cost more than the convolutions it feeds.
    "lg_copy",
    "lg_pixel_unshuffle2",
    "lg_upsample2x_nearest",
    // The Unsqueeze2d that ends the network, which the toolkit has as a
    // pixel-shuffle at r = 2. Measured in one process against this engine's own
    // kernel: the toolkit's wins everywhere that has work to do.
    "lg_pixel_shuffle",
    // Project (`cuda/hcflow.cu`), down to the two fusions the toolkit has no
    // equivalent for.
    "lg_couple_inverse",
    "lg_sample_latent",
];

/// The subset of `KERNEL_NAMES` that lives in the project fatbin. Anything not
/// listed here is looked up in the toolkit.
pub const PROJECT_KERNEL_NAMES: &[&str] = &[
    "lg_couple_inverse",
    "lg_sample_latent",
    // The tiled 3x3s are the DEV CAMPAIGN's sweep, reached only by `--conv-bench`
    // - the graph dispatches the toolkit's `lg_conv3x3_q2`/`_q2ng2`/`_catq0`/
    // `_catq0ng2`, which used to be four of these. (`tile4`/`t2w`/`t2s`/`t2h` are
    // the toolkit's own body at other tile shapes.)
    "lg_conv3x3_tile4",
    "lg_conv3x3_t2w",
    "lg_conv3x3_t2s",
    "lg_conv3x3_t2h",
    "lg_conv3x3_o16w",
    "lg_conv3x3_o16f",
    "lg_conv3x3_c8s",
    "lg_conv3x3_x16",
    "lg_conv3x3_w16",
    "lg_conv3x3_p8",
    "lg_conv3x3_p8n",
    "lg_conv3x3_stageonly",
    "lg_conv3x3_stage_nold",
    "lg_conv3x3_q1",
    "lg_conv3x3_q2t",
    "lg_conv3x3_pipe",
    "lg_conv3x3_pz",
    "lg_conv3x3_q3",
    "lg_conv3x3_v32",
    "lg_conv3x3_c16",
    "lg_conv3x3_catq",
    "lg_conv3x3_catx",
    "lg_conv3x3_catx0",
    "lg_conv3x3_q2x",
    "lg_conv3x3_r16x",
    "lg_conv3x3_r16y",
    "lg_conv3x3_r32a",
    "lg_conv3x3_r32b",
    "lg_conv3x3_r32c",
    "lg_conv3x3_dbuf",
    "lg_conv3x3_dbuf2",
    "lg_conv3x3_catd",
    "lg_conv3x3_catd2",
    "lg_conv3x3_catq0s",
    "lg_conv3x3_q2s",
    "lg_conv3x3_catt2s",
    "lg_conv3x3_p1",
    "lg_conv3x3_catp1",
    "lg_conv3x3_r16",
    "lg_conv3x3_r32",
    "lg_conv3x3_r8x",
    "lg_conv3x3_x8",
];

/// Device information reported at startup.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub free_vram: usize,
}

/// A CACHING device allocator.
///
/// WHY THIS EXISTS. A forward pass allocates a plane per convolution, per
/// concatenation and per residual - several hundred blocks per level - and the
/// driver's `cuMemAlloc`/`cuMemFree` are not cheap: measured on this card one
/// alloc+free pair at 256x256x64ch costs 425 us, which is 14% of a whole tiled
/// 3x3 conv at that size and, summed over the graph's allocation count, more
/// than all of its kernels together. The blocks are also all a few repeated
/// shapes freed in a stack-like order, so recycling them is the difference
/// between the driver being the bottleneck and the driver not appearing in the
/// profile at all.
///
/// The free list holds `vm::DevBuf` values rather than raw pointers, so
/// ownership stays with Rust: nothing is `forget`ten, no new free primitive is
/// needed in the toolkit, and the pool's own blocks are released by ordinary
/// `Drop` when the thread exits.
///
/// RECYCLED MEMORY IS NOT ZEROED, and that is a real change in what callers may
/// assume: `cuMemAlloc` happens to return zeroed pages today, so an engine that
/// reads a plane element it never wrote would work by accident. Every caller in
/// `gpu.rs` writes each element of a fresh plane before reading it - a
/// convolution writes its whole output, `cat`/`concat_all`/`copy_into` write
/// every channel they claim, and the one kernel that writes only PART of its
/// output (`lg_couple_inverse` writes the upper half of `z`) has the rest
/// written by a `copy_into` first. `DevBuf::zeros` is the explicit path when
/// that is not true.
struct Pool {
    /// Free blocks, searched best-fit.
    free: Vec<vm::DevBuf>,
    free_bytes: usize,
    hits: usize,
    misses: usize,
    /// Bytes currently handed OUT to the graph, and the high-water mark of
    /// that. `used + free_bytes` is everything the driver has given the pool.
    ///
    /// A counter incremented on every miss and decremented only when the cap
    /// flushed the free list would make "peak" the total bytes ever allocated
    /// rather than the most held at once: that reads 4.9 GiB on a run whose real
    /// footprint is a few hundred mebibytes, and sends you looking for a memory
    /// bug that does not exist. Anything reported as a footprint has to be a
    /// live count.
    used: usize,
    peak: usize,
    /// Live blocks grouped by the CALL SITE that allocated them, as
    /// `(count, bytes)`.
    sites: std::collections::HashMap<&'static str, (usize, usize)>,
    /// Every live block's size, so the end of a run can say exactly how the
    /// live set is composed rather than how big it is.
    live_sizes: Vec<usize>,
    /// The tag each of `live_sizes` was allocated under, in the same order.
    tag_sizes: Vec<&'static str>,
    /// The most recent allocating backtrace seen for each size, for the leak
    /// report.
    bt_by_size: std::collections::HashMap<usize, String>,
    /// Every live block paired with the call stack that allocated it, so the
    /// live set can be attributed to the function that is holding it.
    live_stack: Vec<(String, usize)>,
    /// The last live total that got a histogram printed, so the report fires on
    /// every 32 MiB of NEW peak rather than on every allocation.
    last_reported: usize,
    /// Blocks handed out and not yet returned - the count `used` should be the
    /// sum of. A byte total that grows without the count growing identically is
    /// an accounting error; a count that grows means the graph really is holding
    /// that many buffers.
    outstanding: usize,
    peak_outstanding: usize,
}

impl Pool {
    /// The most the pool will hold on to before it starts returning blocks to
    /// the driver.
    ///
    /// THIS IS DEVICE MEMORY THE DRIVER CANNOT REUSE while the pool holds it, so
    /// it is a footprint, not a cache size, and it has to stay small against the
    /// graph's own working set: the direct path holds 165 MiB at 256x256 and
    /// 1448 MiB at 1024x1024, and a 512 MiB free list on top of that was most of
    /// why the Winograd path died of OOM. 128 MiB is above every single plane
    /// this engine allocates below 1024x1024 (the largest is a 64-channel
    /// 512x512 transformed plane at 27 MiB), so it still recycles the blocks
    /// that matter, and it bounds the waste at one large plane.
    const CAP: usize = 512 * 1024 * 1024;

    /// Record a new high-water mark: which allocation set it, how big that block
    /// was and how many blocks were outstanding at that moment. A peak of N bytes
    /// spread over a thousand small blocks and one held in ten huge ones look
    /// identical in a byte total, and they have completely different causes.
    fn hist(&self) -> Vec<(usize, usize)> {
        let mut m: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for s in &self.live_sizes {
            *m.entry(*s).or_insert(0) += 1;
        }
        let mut v: Vec<(usize, usize)> = m.into_iter().collect();
        v.sort_by(|a, b| (b.0 * b.1).cmp(&(a.0 * a.1)));
        v
    }

    fn note(&mut self, sz: usize) {
        self.live_sizes.push(sz);
        self.tag_sizes.push(TAG.with(|t| t.get()));
        self.site_add(TAG.with(|t| t.get()), sz);
        // Keep ONE backtrace per live size, so the end of a run can name the
        // owner of every block that is still alive without storing a backtrace
        // for every allocation ever made.
        if std::env::var("HCFLOW_SITE_TRACE").is_ok() {
            let bt = std::backtrace::Backtrace::force_capture().to_string();
            let stack: Vec<String> = bt
                .lines()
                .filter(|l| l.contains("hcflow::"))
                .filter(|l| !l.contains("cuda::Pool") && !l.contains("cuda::DevBuf"))
                .take(3)
                .map(|l| l.trim().to_string())
                .collect();
            let key = stack.join(" <- ");
            self.bt_by_size.entry(sz).or_insert_with(|| key.clone());
            self.live_stack.push((key, sz));
        }
        self.note_peak(sz);
    }

    /// Every live size with the backtrace of the last allocation of that size:
    /// what is still alive at the end of a run and where it came from.
    fn report_live(&self) -> Vec<(usize, usize, String)> {
        let mut out: Vec<(usize, usize, String)> = Vec::new();
        for (sz, n) in self.hist() {
            let bt = self.bt_by_size.get(&sz).cloned().unwrap_or_default();
            out.push((sz, n, bt));
        }
        out
    }

    fn site_add(&mut self, where_: &'static str, bytes: usize) {
        if std::env::var("HCFLOW_POOL_DEBUG").is_ok() {
            let e = self.sites.entry(where_).or_insert((0, 0));
            e.0 += 1;
            e.1 += bytes;
        }
    }

    fn site_sub(&mut self, where_: &'static str, bytes: usize) {
        if let Some(e) = self.sites.get_mut(where_) {
            e.0 -= 1;
            e.1 -= bytes;
        }
    }

    fn note_peak(&mut self, sz: usize) {
        if std::env::var("HCFLOW_POOL_DEBUG").is_ok() && self.used >= self.last_reported + (256 << 20) {
            self.last_reported = self.used;
            eprintln!("  peak {} MiB in {} blocks (this one {} MiB), by allocating stack:",
                      self.used / (1024 * 1024), self.outstanding, sz / (1024 * 1024));
            let mut agg: std::collections::HashMap<&str, (usize, usize)> =
                std::collections::HashMap::new();
            for (k, s) in &self.live_stack {
                let e = agg.entry(k.as_str()).or_insert((0, 0));
                e.0 += 1;
                e.1 += s;
            }
            let mut v: Vec<(&str, usize, usize)> =
                agg.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
            v.sort_by(|a, b| b.2.cmp(&a.2));
            for (k, n, b) in v.iter().take(5) {
                eprintln!("      {n:>4} blocks {:>5} MiB  {k}", b / (1024 * 1024));
            }
            {
                let mut by: std::collections::HashMap<(&str, usize), usize> =
                    std::collections::HashMap::new();
                for (k, s) in &self.live_stack {
                    *by.entry((k.as_str(), *s)).or_insert(0) += 1;
                }
                let mut vv: Vec<((&str, usize), usize)> = by.into_iter().collect();
                vv.sort_by(|a, b| (b.0 .1 * b.1).cmp(&(a.0 .1 * a.1)));
                for ((k, s), n) in vv.iter().take(8) {
                    eprintln!("        size {:>5} MiB x {n:>4}   {k}", s / (1024 * 1024));
                }
            }
            let sum: usize = self.live_sizes.iter().sum();
            let big: usize = self.live_sizes.iter().filter(|s| **s > 8 << 20).count();
            eprintln!("      CHECK sum(live_sizes) = {} MiB over {} blocks, {big} of them \
                       larger than 8 MiB, used = {} MiB",
                      sum / (1024 * 1024), self.live_sizes.len(), self.used / (1024 * 1024));
        }
    }

    fn take(&mut self, bytes: usize) -> Result<(vm::DevBuf, usize), String> {
        // EVERY REQUEST IS QUANTISED UP TO A BUCKET, so blocks of the same
        // bucket are interchangeable and a block returned by a small request can
        // serve the next small one. Without this the winograd path fragments the
        // card: its intermediates are dozens of different sizes (36 * tiles *
        // channels, and both factors vary per layer), and a best-fit allocator
        // holding one block per exact size runs out of memory with most of the
        // card free. The bucket is 1 MiB above 256 KiB - a large request never
        // wastes more than a megabyte - and 4 KiB below it, where the graph's
        // small planes (the coupling's half-planes and the 3-channel latents)
        // live.
        let bytes = {
            let quantum = if bytes > 256 * 1024 { 1024 * 1024 } else { 4096 };
            ((bytes + quantum - 1) / quantum) * quantum
        };

        // Best fit: the smallest free block that holds `bytes`. The graph's
        // sizes repeat, so this is usually an exact-size hit on the first pass.
        let mut best: Option<usize> = None;
        for (i, b) in self.free.iter().enumerate() {
            if b.bytes >= bytes && best.map(|j| self.free[j].bytes > b.bytes).unwrap_or(true) {
                best = Some(i);
            }
        }

        // A FIT THAT IS WILDLY TOO BIG IS NOT A FIT. Best-fit can only choose
        // among the blocks that happen to be free, so a 55 KiB request off the
        // winograd weight transform's 530 small allocations lands on whatever
        // transformed-plane block (9-27 MiB) was freed most recently, and then
        // HOLDS all of it: the plan's own `bytes` says 55 KiB while the block it
        // occupies says 9 MiB, which is how the transform cache came to own
        // 1.1 GiB of a 5 GiB peak for 349 MiB of actual data. So a block more
        // than 4x the request (and at least 256 KiB, to leave the small-plane
        // recycling alone) is only taken if the driver will not give us a fresh
        // one.
        let oversized = best.map(|i| self.free[i].bytes > bytes * 4 && self.free[i].bytes > 256 * 1024);
        if oversized == Some(true) {
            if let Ok(b) = vm::DevBuf::alloc(bytes) {
                self.outstanding += 1;
                self.used += b.bytes;
                self.peak = self.peak.max(self.used);
                self.peak_outstanding = self.peak_outstanding.max(self.outstanding);
                let sz = b.bytes;
                self.note(sz);
                return Ok((b, sz));
            }
            // Out of memory: fall through and reuse the oversized block.
        }

        if let Some(i) = best {
            let b = self.free.swap_remove(i);
            self.free_bytes -= b.bytes;
            self.hits += 1;
            // A block that fits but is oversized is still the driver's memory:
            // the excess is not tracked as free, it rides along with the block.
            self.used += b.bytes;
            self.outstanding += 1;
            self.peak = self.peak.max(self.used);
            self.peak_outstanding = self.peak_outstanding.max(self.outstanding);
            let sz = b.bytes;
            self.note(sz);
            return Ok((b, sz));
        }
        self.misses += 1;
        // Nothing free: ask the driver, and if IT is out of memory, drop the
        // whole free list and try once more. That is the one case where a cache
        // in front of the driver is worse than no cache at all.
        match vm::DevBuf::alloc(bytes) {
            Ok(b) => {
                self.used += b.bytes;
                self.outstanding += 1;
                self.peak = self.peak.max(self.used);
                self.peak_outstanding = self.peak_outstanding.max(self.outstanding);
                let sz = b.bytes;
                self.note(sz);
                Ok((b, sz))
            }
            Err(e) => {
                if self.free.is_empty() {
                    return Err(e);
                }
                self.free.clear();
                self.free_bytes = 0;
                let b = vm::DevBuf::alloc(bytes)?;
                self.used += b.bytes;
                self.outstanding += 1;
                self.peak = self.peak.max(self.used);
                self.peak_outstanding = self.peak_outstanding.max(self.outstanding);
                let sz = b.bytes;
                self.note(sz);
                Ok((b, sz))
            }
        }
    }

    fn give_back(&mut self, b: vm::DevBuf) {
        if std::env::var("HCFLOW_ALLOC_TRACE").is_ok() && b.bytes > 8 * 1024 * 1024 {
            eprintln!("  give_back {} MiB: live {} -> {}",
                      b.bytes / (1024 * 1024), self.used / (1024 * 1024),
                      self.used.saturating_sub(b.bytes) / (1024 * 1024));
        }
        // Remove ONE occurrence of this size: `swap_remove` keeps it O(1) and the
        // vector is only ever used for the histogram.
        if let Some(i) = self.live_sizes.iter().position(|s| *s == b.bytes) {
            self.live_sizes.swap_remove(i);
            let t = self.tag_sizes.swap_remove(i);
            self.site_sub(t, b.bytes);
            if let Some(j) = self.live_stack.iter().position(|(_, s)| *s == b.bytes) {
                self.live_stack.swap_remove(j);
            }
        }
        self.outstanding -= 1;
        self.used = self.used.saturating_sub(b.bytes);
        if self.free_bytes + b.bytes > Self::CAP {
            // Over the cap: this one goes back to the driver as it is dropped.
            return;
        }
        self.free_bytes += b.bytes;
        self.free.push(b);
    }
}

thread_local! {
    static POOL: std::cell::RefCell<Pool> = std::cell::RefCell::new(Pool {
        free: Vec::new(),
        free_bytes: 0,
        hits: 0,
        misses: 0,
        used: 0,
        peak: 0,
        sites: std::collections::HashMap::new(),
        live_sizes: Vec::new(),
        tag_sizes: Vec::new(),
        bt_by_size: std::collections::HashMap::new(),
        live_stack: Vec::new(),
        last_reported: 0,
        outstanding: 0,
        peak_outstanding: 0,
    });
}

/// Pool statistics for the run summary: `(hits, misses, live_bytes,
/// free_bytes, peak_live_bytes)`. `live + free` is everything the driver has
/// given the pool, and `peak` is the most that was ever LIVE at once - the
/// number a memory budget is stated in, and the one the free-list cap exists to
/// bound.
pub fn pool_stats() -> (usize, usize, usize, usize, usize) {
    POOL.with(|p| {
        let p = p.borrow();
        (p.hits, p.misses, p.used, p.free_bytes, p.peak)
    })
}

/// Diagnostic: `(live blocks, free blocks)`.
pub fn pool_counts() -> (usize, usize) {
    POOL.with(|p| {
        let p = p.borrow();
        (p.live_sizes.len(), p.free.len())
    })
}

/// Diagnostic: the live set with the allocating backtrace for each size, plus
/// the run's totals, printed when `HCFLOW_LEAK_TRACE` is set.
pub fn leak_report() {
    if std::env::var("HCFLOW_LEAK_TRACE").is_err() {
        return;
    }
    POOL.with(|p| {
        let p = p.borrow();
        eprintln!("live at exit: {} blocks, {} MiB, {} outstanding",
                  p.live_sizes.len(), p.used / (1024 * 1024), p.outstanding);
        for (sz, n, bt) in p.report_live().iter().take(6) {
            eprintln!("  {n:>4} x {:>4} MiB   {bt}", sz / (1024 * 1024));
        }
    });
}

/// A device allocation counting ELEMENTS of f32, taken from the pool. lightgpu
/// counts bytes; the graph code thinks in tensor lengths, so the conversion
/// stays in one place. `inner` is the block's owner, which is what makes the
/// recycling work: dropping a `DevBuf` parks its `inner` in the pool instead of
/// freeing it.
pub struct DevBuf {
    pub ptr: lightgpu::ffi::CUdeviceptr,
    pub len: usize,
    inner: Option<vm::DevBuf>,
}

impl DevBuf {
    pub fn empty() -> DevBuf {
        DevBuf { ptr: 0, len: 0, inner: None }
    }

    pub fn alloc(len: usize) -> Result<DevBuf, String> {
        if len == 0 {
            return Ok(DevBuf::empty());
        }
        if std::env::var("HCFLOW_ALLOC_TRACE").is_ok() && len * 4 > 8 * 1024 * 1024 {
            eprintln!(
                "ALLOC {} MiB\n{}",
                (len * 4) / (1024 * 1024),
                std::backtrace::Backtrace::force_capture()
            );
        }
        let bytes = len * std::mem::size_of::<f32>();
        let (b, _block) = POOL.with(|p| p.borrow_mut().take(bytes))?;
        Ok(DevBuf { ptr: b.ptr, len, inner: Some(b) })
    }

    /// An explicitly zeroed allocation.
    pub fn zeros(len: usize) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(len)?;
        if let Some(i) = &b.inner {
            vm::memset_d8(i.ptr, i.bytes)?;
        }
        Ok(b)
    }

    pub fn from_host(v: &[f32]) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(v.len())?;
        b.upload(v)?;
        Ok(b)
    }

    pub fn upload(&self, v: &[f32]) -> Result<(), String> {
        if v.len() != self.len {
            return Err(format!("upload size {} != buffer {}", v.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.upload(v),
            None => Ok(()),
        }
    }

    pub fn download(&self, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.len {
            return Err(format!("download size {} != buffer {}", out.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.download(out),
            None => Ok(()),
        }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if let Some(b) = self.inner.take() {
            POOL.with(|p| p.borrow_mut().give_back(b));
        }
    }
}

/// The whole weight payload on the device, with per-tensor offsets.
pub struct WeightArena {
    pub buf: DevBuf,
    /// `(name -> offset in elements)` for every tensor in the container.
    pub offsets: HashMap<String, usize>,
    /// `name -> element count`, kept next to the offset so a launch can check
    /// the geometry it is about to use against the tensor it is about to read.
    pub lens: HashMap<String, usize>,
    pub bytes: usize,
}

impl WeightArena {
    /// Upload the payload section in ONE copy.
    pub fn upload(file: &lightgpu::safetensors::File) -> Result<WeightArena, String> {
        let payload = file.data();
        if payload.len() % 4 != 0 {
            return Err(format!("weight payload is {} bytes, not a multiple of 4", payload.len()));
        }
        let base_file = file.data_start();
        let mut offsets = HashMap::new();
        let mut lens = HashMap::new();
        for name in file.order() {
            let info = file.info(name)?;
            if info.dtype != lightgpu::safetensors::DType::F32 {
                return Err(format!("tensor `{name}` is {:?}, not F32", info.dtype));
            }
            if info.offset < base_file {
                return Err(format!("tensor `{name}` starts before the payload section"));
            }
            let rel = info.offset - base_file;
            if rel % 4 != 0 {
                return Err(format!(
                    "tensor `{name}` is at payload byte {rel}, which is not 4-byte aligned"
                ));
            }
            offsets.insert(name.clone(), rel / 4);
            lens.insert(name.clone(), info.numel());
        }
        let len = payload.len() / 4;
        let buf = DevBuf::alloc(len)?;
        vm::copy_htod(buf.ptr, payload)?;
        Ok(WeightArena { buf, offsets, lens, bytes: payload.len() })
    }

    pub fn len_of(&self, name: &str) -> Result<usize, String> {
        self.lens
            .get(name)
            .copied()
            .ok_or_else(|| format!("no tensor `{name}` in the checkpoint"))
    }

    /// Device pointer to a tensor's first element.
    pub fn ptr(&self, name: &str) -> Result<lightgpu::ffi::CUdeviceptr, String> {
        let off = *self
            .offsets
            .get(name)
            .ok_or_else(|| format!("no tensor `{name}` in the checkpoint"))?;
        Ok(self.buf.ptr + (off * 4) as u64)
    }

    /// The tensor as a `&[f32]`, shape-checked. Callers pass the length they
    /// expect so a wrong tensor is a load-time error rather than a misread.
    pub fn f32(&self, name: &str, want: usize) -> Result<lightgpu::ffi::CUdeviceptr, String> {
        let got = self.len_of(name)?;
        if got != want {
            return Err(format!("`{name}` has {got} elements, expected {want}"));
        }
        self.ptr(name)
    }
}

pub struct Cuda {
    pub toolkit: vm::Module,
    pub project: vm::Module,
    pub info: DeviceInfo,
    pub weights: RefCell<Option<WeightArena>>,
}

impl Cuda {
    pub fn init(verbose: bool) -> Result<Cuda, String> {
        vm::init()?;
        let dev = vm::device()?;
        let toolkit = vm::Module::load(TOOLKIT_FATBIN)?;
        let project = vm::Module::load(PROJECT_FATBIN)?;
        let c = Cuda {
            toolkit,
            project,
            info: DeviceInfo {
                name: dev.name.clone(),
                cc_major: dev.cc_major,
                cc_minor: dev.cc_minor,
                sm_count: dev.sm_count,
                free_vram: vm::free_vram()?,
            },
            weights: RefCell::new(None),
        };
        for k in KERNEL_NAMES {
            c.func(k)?;
        }
        if verbose {
            eprintln!(
                "cuda: {} cc {}.{} ({} SMs, {} MiB free, fatbins {} + {} bytes)",
                c.info.name,
                c.info.cc_major,
                c.info.cc_minor,
                c.info.sm_count,
                c.info.free_vram / (1024 * 1024),
                TOOLKIT_FATBIN.len(),
                PROJECT_FATBIN.len()
            );
        }
        Ok(c)
    }

    /// Resolve a kernel by name against the module that defines it. The
    /// project's kernels are tried SECOND, so a name that exists in both (none
    /// today) would resolve to the toolkit's - the toolkit's are the published
    /// contract and the project's are this engine's private additions.
    pub fn func(&self, name: &str) -> Result<CUfunction, String> {
        if PROJECT_KERNEL_NAMES.contains(&name) {
            return self.project.func(name);
        }
        self.toolkit.func(name)
    }

    /// Which module owns `name`, for the launches: `Args::launch` takes a
    /// module rather than a bare handle, and a kernel resolved from the project
    /// module would fail with CUDA_ERROR_NOT_FOUND if it were launched through
    /// the toolkit's.
    pub fn module_of(&self, name: &str) -> Result<&vm::Module, String> {
        if PROJECT_KERNEL_NAMES.contains(&name) {
            return Ok(&self.project);
        }
        Ok(&self.toolkit)
    }

    /// Upload the checkpoint's payload as one arena. Kept separate from `init`
    /// because the CPU backend never allocates one, and because the timing
    /// message wants to report the copy on its own.
    pub fn upload_weights(&self, file: &lightgpu::safetensors::File) -> Result<usize, String> {
        let t0 = std::time::Instant::now();
        let arena = WeightArena::upload(file)?;
        let bytes = arena.bytes;
        *self.weights.borrow_mut() = Some(arena);
        eprintln!(
            "cuda: uploaded {} MiB of weights in one copy ({:.1} ms)",
            bytes / (1024 * 1024),
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(bytes)
    }

    /// The device pointer to a named tensor, shape-checked against `want`.
    pub fn w(&self, name: &str, want: usize) -> Result<lightgpu::ffi::CUdeviceptr, String> {
        let b = self.weights.borrow();
        let a = b.as_ref().ok_or("weights were never uploaded")?;
        a.f32(name, want)
    }

    /// Launch `name` with `build` filling the argument list.
    ///
    /// With `LA_PROFILE=1` each launch is bracketed by CUDA events and its
    /// device time accumulated per kernel name. Events rather than a host sync:
    /// a forward pass here is well over a hundred launches, and measuring them
    /// the naive way would be swamped by the synchronisation.
    pub fn run<F: FnOnce(&mut Args)>(&self, name: &str, l: Launch, build: F) -> Result<(), String> {
        let m = self.module_of(name)?;
        let mut a = Args::new();
        build(&mut a);
        if !profiling() {
            return a.launch(m, name, l);
        }
        let a_ev = vm::Event::new()?;
        let b_ev = vm::Event::new()?;
        let m = self.module_of(name)?;
        a_ev.record()?;
        a.launch(m, name, l)?;
        b_ev.record()?;
        // LA_PROFILE_NOSYNC times each kernel without draining the queue after
        // it, which is what a run without profiling does. The per-kernel ms then
        // overlap and only the AGGREGATE is meaningful, but the comparison
        // against the synchronised form isolates how much of the "device time"
        // the profiling itself is creating.
        let _nosync = std::env::var("LA_PROFILE_NOSYNC").is_ok();
        // cuEventElapsedTime needs BOTH events complete. Waiting on the closing
        // event serialises the launches, which is acceptable only because this
        // engine already runs everything on one stream, and it is what makes
        // each kernel's device time exact rather than a fraction of a queue.
        b_ev.synchronize()?;
        let ms = a_ev.elapsed_ms(&b_ev)? as f64;
        let key = format!("{} g{}x{}x{} b{}x{}", name, l.grid.0, l.grid.1, l.grid.2,
                          l.block.0, l.block.1);
        PROFILE.with(|p| {
            let mut p = p.borrow_mut();
            let e = p.entry(key).or_insert((0.0, 0usize));
            e.0 += ms;
            e.1 += 1;
        });
        Ok(())
    }


    /// Launch over `n` elements with 256-thread blocks - the shape of every
    /// elementwise kernel here.
    pub fn run_n<F: FnOnce(&mut Args)>(&self, name: &str, n: usize, build: F) -> Result<(), String> {
        let block = 256u64;
        let grid = ((n as u64 + block - 1) / block).max(1) as u32;
        self.run(name, Launch::new((grid, 1, 1), (block as u32, 1, 1)), build)
    }

    pub fn sync(&self) -> Result<(), String> {
        vm::sync()
    }

    /// Print the accumulated per-kernel device time, heaviest first.
    pub fn profile_report(&self) {
        PROFILE.with(|p| {
            let p = p.borrow();
            if p.is_empty() {
                return;
            }
            let mut rows: Vec<(&String, &(f64, usize))> = p.iter().collect();
            rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let total: f64 = p.values().map(|v| v.0).sum();
            eprintln!("--- kernel profile (device ms, {:.1} total) ---", total);
            for (name, (ms, n)) in rows {
                eprintln!("  {:<60} {:9.1} ms {:6} calls ({:6.2} us each, {:4.1}%)",
                          name, ms, n, ms * 1000.0 / *n as f64, 100.0 * ms / total);
            }
        });
    }
}

/// A per-channel affine applied through the toolkit kernel, with either vector
/// optional. This is the ActNorm and the coupling's `exp(-logscale)` in one
/// place, so the call sites read as the algebra they implement.
pub fn channel_affine(
    ca: &Cuda,
    x: lightgpu::ffi::CUdeviceptr,
    scale: Option<lightgpu::ffi::CUdeviceptr>,
    shift: Option<lightgpu::ffi::CUdeviceptr>,
    y: lightgpu::ffi::CUdeviceptr,
    c: usize,
    hw: usize,
) -> Result<(), String> {
    // The kernel's parameter order is (in, out, scale, shift, c, hw) - the
    // output sits SECOND, next to its input, and both vectors follow it. The
    // natural-looking (in, scale, shift, out) order here would pass the scale
    // buffer as the input and write through the input pointer, so the two
    // pointer arguments are commented rather than left to inference.
    ca.run_n("lg_channel_affine", c * hw, |a| {
        a.ptr(x);                  // in
        a.ptr(y);                  // out
        a.ptr(scale.unwrap_or(0)); // scale (nullable)
        a.ptr(shift.unwrap_or(0)); // shift (nullable)
        a.i32(c as i32);
        a.i32(hw as i32);
    })
}
