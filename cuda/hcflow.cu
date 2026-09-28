// HCFlow's own kernels.
//
// Everything the flow needs is in the shared `lightgpu` toolkit, with two
// exceptions, both of them fusions rather than new arithmetic: the coupling's
// inversion, which folds its split-into-shift-and-scale, its
// `0.318 * atan(2 * scale)` and the inversion of the lower half of `z` into one
// pass, and the conditional flow's latent sampler. Each of them would otherwise
// be three launches and two extra planes per call, and there are 52 couplings per
// level.
//
// The Unsqueeze2d inverse that ends the network USED TO BE HERE and is now the
// toolkit's `lg_pixel_shuffle` at r = 2: an earlier comment on this file claimed
// the toolkit had no pixel-shuffle kernel, which stopped being true when that one
// was promoted, and the two forms were then measured against each other in one
// process on sixteen geometries, both of the graph's own call sites among them.
// At those two the two forms are a wash (4.1 vs 4.3 us and 6.6 vs 6.0 us, on a
// 310 ms pass), but the project's form collapses on the flatter, wider shapes
// - 0.40x at 128x32x32, 0.61x at 64x32x32 - because it loops over channels
// inside the thread rather than putting the channel on the grid, and it is gone.
// See `gpu::unsqueeze2d` for the geometry and the numbers.

// The coupling's fused inversion.
//
// The reference coupling runs its FCN on `cat([z1, cond])`, gives a 2*half
// channel plane `h`, and then (AffineCouplings.Affine with the Affine3shift
// logscale the SR checkpoints use):
//
//     shift = h[0::2]            scale = h[1::2]
//     logscale = 0.318 * atan(2 * scale)
//     y = x * exp(logscale) + shift        (forward)
//     x = (y - shift) * exp(-logscale)     (reverse)
//
// applied to the LOWER half of z only, with the upper half passing through
// untouched. Doing that as separate kernels means materialising shift, scale
// and exp(-logscale) planes (three passes over 2*half channels) plus the
// subtraction, and it is the innermost op of 63 couplings per level, so it is
// fused here: each thread reads one element of h's even and odd planes and the
// matching element of z's lower half, and writes the result in place.
//
// `0.318` is the reference's literal (`logscale = 0.318 * atan(2 * scale)` in
// modules/AffineCouplings.py), kept as written rather than folded to a shorter
// constant so the CPU twin can use the identical expression.
//
// `z_in` and `z_out` are SEPARATE pointers rather than one in-place buffer: the
// caller has already written `z1` (channels [0, half)) into `z_out`, and the
// second half of `z_out` holds nothing until this kernel fills it. An in-place
// version would read its own uninitialised output for every element of the
// lower half of `z`.
//
// Geometry: one thread per element of the (half, h, w) second half; grid
// (ceil(hw/256), n2, 1), block (256, 1, 1), which matches `lg_relu`'s
// elementwise shape.
extern "C" __global__ void lg_couple_inverse(
    const float *__restrict__ h,
    const float *__restrict__ z_in, float *__restrict__ z_out,
    int half, int hw)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (p >= hw) return;
    const float shift = h[(2 * c) * hw + p];
    const float scale = h[(2 * c + 1) * hw + p];
    const float logscale = 0.318f * atanf(2.0f * scale);
    // FORWARD is z2 = (z2 + shift) * exp(logscale) - the shift is applied
    // BEFORE the scale - so the reverse is `z2 * exp(-logscale) - shift`, with
    // the multiplication first. That order is the reference's, and it is not
    // interchangeable with `(z2 - shift) * exp(-logscale)`: the two differ by
    // shift * (1 - exp(-logscale)), which a 63-coupling chain amplifies.
    z_out[(half + c) * hw + p] = z_in[(half + c) * hw + p] * __expf(-logscale) - shift;
}

// The conditional flow's latent sampler.
//
// `ConditionalFlow.reverse_flow` runs its FCN on the 128-channel conditional
// feature, gets a 2*cond_in-channel plane, and reads it with
// `split_feature(h, "cross")` - mean = h[:, 0::2], logs = h[:, 1::2] - then
// draws `z = mean + exp(logs) * eps` with `eps ~ N(0, eps_std^2)` in the
// reference's own `GaussianDiag.sample`.
//
// The engine passes a UNIT normal eps and folds eps_std into the log-scale
// instead (`z = mean + exp(eps_std * logs) * eps`), which is the same
// distribution but stays well defined at eps_std = 0 and makes a run
// reproducible from a seed without also encoding the temperature. `eps == NULL`
// means "no noise" and is the zero sample.
//
// Doing this on the host would be a download of the whole 2*cond_in plane and an
// upload of the result - 42 channels at 512x512 is 43 MiB each way for the top
// level, once per level per image.
//
// Geometry: one thread per element of the OUTPUT (cond_in, h, w); grid
// (ceil(hw/256), cond_in, 1), block (256, 1, 1).
extern "C" __global__ void lg_sample_latent(
    const float *__restrict__ h, const float *__restrict__ eps,
    float *__restrict__ z, int hw, float eps_std)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (p >= hw) return;
    const float mean = h[(2 * c) * hw + p];
    const float logs = h[(2 * c + 1) * hw + p];
    const float e = (eps == NULL) ? 0.0f : eps[c * hw + p];
    z[c * hw + p] = mean + __expf(eps_std * logs) * e;
}


// ---------------------------------------------------------------------------
// THE 3x3 CEILING ON THIS PART, measured, so it is not re-derived (2026 note)
// ---------------------------------------------------------------------------
//
// Every direct tiled 3x3 in this file lands at 2.5-2.8 TFLOP/s on the two
// shapes that dominate the runtime (64->64 and 64->32 at 512x512), against the
// 8.9 TFLOP/s a pure-FMA microbenchmark reaches on the same card. The
// following were measured and are NOT the cause:
//
//   * shared-memory operand count: replacing BOTH shared operands of the inner
//     FFMA with constants (inner loop LDS = 0) changed the time by <1%;
//   * instruction mix: the same experiment with 38% FMA instead of 26%, <1%;
//   * registers/occupancy: `__launch_bounds__(128, 8)` (64 registers) and
//     `(128, 10)` (48), and a halved shared tile with CIM = 2 so eight blocks
//     actually fit, all within 2%;
//   * launch order: output channels on blockIdx.x instead of blockIdx.z, so the
//     blocks sharing an input tile run consecutively (see `lg_conv3x3_c16`), <1%;
//   * barriers: deleting the two `__syncthreads()` per channel tile made it
//     SLOWER;
//   * bank conflicts: the staged stride rule below removed a measured 2-way
//     conflict and bought 11%, and that is spent;
//   * the card is at 1835-1860 MHz and ~180 W during these runs, i.e. not
//     thermally throttled;
//   * a synthetic loop with the same operand pattern (two shared operands per
//     FFMA) and the same 16 accumulators reaches 8.9 TFLOP/s at 32 blocks/SM,
//     so the pattern is not inherently slow.
//
// What remains is warp-level stall behaviour that the tools here cannot see
// (`ncu` is installed but this user cannot read performance counters:
// ERR_NVGPUCTRPERM), and the measured issue rate - about 0.3 dispatched
// instructions per SM per cycle against a capacity of 4 - says the warps are
// stalled roughly 90% of the time on something not yet identified. A 5x margin
// in arithmetic (cuDNN reaches 5.1 TFLOP/s on these same shapes) is therefore
// still on the table, and finding it needs counter access.
//
// Winograd is NOT the answer: it was implemented, verified (rel 1.8e-6) and is
// slower (1.4-2.0 TFLOP/s), because its transforms cost 648 flops per (tile,
// channel) against a 16-output tile while the GEMM they replace saves only 4x
// on 9 flops.

// ---------------------------------------------------------------------------
// A 3x3 conv for THIS engine's shapes: the toolkit's tiled kernel re-tiled
// ---------------------------------------------------------------------------
//
// WHY A SECOND TILED 3x3. `lg_conv3x3_tile` stages CI = 8 input channels at a
// time, which needs sh[8][10][130] = 41,600 bytes and puts the kernel at ONE
// block per SM - eight warps, 12.5% occupancy - and its weight tile is laid out
// [oc][ci][tap], so the inner loop's eight weight reads per (ci, ky, kx) are
// eight separate scalar shared loads. The kernel is therefore issue-bound on the
// LSU: 108 shared loads against 288 FMAs per channel-tile, a ratio of 2.7:1
// where the hardware offers 4:1. Measured, it runs the trunk's 64->64 3x3 at
// 1372 GFLOP/s (64x64: 648), i.e. 15% of the GTX 1080's fp32 peak.
//
// This kernel changes three things and nothing else about the algorithm:
//
//   * CI = 4, so the staged input tile is 20,800 bytes and TWO blocks fit per
//     SM (16 warps), which is what hides the shared-memory latency;
//   * the weight tile is transposed to [tap][ci][oc], so for a fixed (tap, ci)
//     the eight output-channel weights are CONTIGUOUS and the inner loop loads
//     them as two `float4`s instead of eight scalars - the weight-panel reads
//     fall from 72 to 18 per channel tile;
//   * the same (128 x 8) output tile, the same (ci, ky, kx) accumulation order
//     and the same zero-filled halo, so the two kernels agree to the last bit
//     and `--prim-test` checks whichever one the dispatch runs.
//
// Loads per channel tile fall from 108 to 9*(4 + 2) = 54 against the same 288
// FMAs.
//
// Geometry: grid = (ceil(wd/128), ceil(h/8), ceil(c_out/8)), block = (32,8,1).
// act: 0 none, 1 relu, 2 leaky relu with `act_p` as the slope.
// The tile shape is a TEMPLATE parameter, not a constant: the same 3x3 body at
// TW = 128 wastes half of every block at a 64-wide input (the block computes 128
// columns and the bounds check discards 64 of them - measured: 1004 GFLOP/s at
// 64->64@64x64 against 1869 at 128x128, a factor of exactly two), and a smaller
// tile also raises the blocks per SM, which is what hides the staging loads.
// Four shapes are instantiated and `--conv-bench` measures them all; the
// dispatch in `gpu.rs` picks by the input width.
//
// All of them share the body below, so they agree bit for bit: same (ci, ky, kx)
// accumulation order, same zero-filled halo, same bias-then-activation epilogue.
//
// WHERE THE TIME ACTUALLY GOES - three probes, and the answer is NOT arithmetic.
// Every shape and every restructuring of these kernels used to land at 2.7-2.8
// TFLOP/s on the 64->64 and 64->32 trunk shapes (x16, x8, t2s, c8s, o16w, o16f,
// r16, r32, r8x, winograd) while a pure-FMA microbenchmark with no memory
// operands reaches 8.9 TFLOP/s, and the structural experiments all came back
// under 2%: replacing BOTH shared-memory operands of the inner FFMA with
// constants, changing the FMA fraction of the instruction stream from 26% to
// 38%, and forcing 8 or 10 blocks per SM through `__launch_bounds__` each moved
// the time by under 2%. Those experiments are all consistent and they were all
// measuring the wrong thing. The three that settle it, on the same kernel at
// 64->64@512x512 in one process with repeats agreeing to 0.2%:
//   x16        7.05 ms   the kernel as it is
//   `stage`    6.19 ms   same staging, same barriers, same epilogue, inner loop
//                        cut to ONE multiply per output channel (1/9 of the FMAs)
//   `nold`     1.14 ms   everything `stage` has, but the value that would come
//                        from DRAM is computed from the indices instead
// So removing 8/9 of the arithmetic saves 0.86 ms and removing the global load
// saves 5.04 ms OF 7.05: THE KERNEL IS LOAD-BOUND, and all the shared stores,
// both barriers and the epilogue together cost 1.15 ms. The loads are not
// bandwidth-starved either - staging the same bytes in isolation with the same
// grid runs at ~260 GB/s, essentially the card's 320 GB/s peak, against ~137
// GB/s inside the kernel - they are EXPOSED, because the staging loop is LDG,
// STS, LDG, STS... and each shared store waits on its own load with nothing to
// cover it.
// CONSEQUENCES, all measured: prefetching the next channel tile INTO REGISTERS
// loses 30% (`lg_conv3x3_pipe`; `q2` is already at its register cap), while
// prefetching it into a SECOND SHARED BUFFER is roughly free in registers and
// was worth 25-30% on one machine state and nothing on another (see
// `lg_dbuf_body`, which is why the double-buffered family is a measured
// alternative rather than the default). Putting the output-channel group on
// blockIdx.x instead of
// blockIdx.z is worth 8-11%, because it makes the blocks that share an input
// tile run consecutively so L2 serves the re-read. Cutting the re-read by
// raising the output-channel group per block loses, because `acc[OC][TPX]` and
// the weight vector scale with OC. cuDNN does this shape in 2.412 ms = 8015
// GFLOP/s, so a direct tiled 3x3 is not inherently stuck at 2.8 TFLOP/s - the
// remaining lever is the ~8x re-read of the input plane, not the arithmetic.
template <int TBX, int TBY, int TPX, int OCM = 8, int CIM = 4, int ZFIRST = 1,
          bool PREZERO = false, int ZSWAP = 0, int NG = 1>
__device__ __forceinline__ void lg_tile_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p,
    const float *__restrict__ i1 = nullptr, const float *__restrict__ i2 = nullptr,
    const float *__restrict__ i3 = nullptr, const float *__restrict__ i4 = nullptr,
    const float *__restrict__ i5 = nullptr,
    int g1 = 0, int g2 = 0, int g3 = 0, int g4 = 0, int g5 = 0, int ng = 1)
{
    constexpr int TW = TBX * TPX;   // output columns per tile
    constexpr int TH = TBY;         // output rows per tile
    constexpr int CI = CIM;         // input channels per staged tile
    constexpr int OC = OCM;         // output channels per tile, accumulators per thread
    // Staged width, with the halo, rounded so that the ROW STRIDE does not put
    // two rows of a warp into the same banks. A warp covers two rows of the
    // staged tile (TBX = 16 threads to a row), and the row stride is SW floats,
    // so with SW = TW + 2 = 66 the second row sits 2 banks from the first and
    // the two 16-wide reads overlap in 14 banks - a two-way conflict on every
    // input load in the inner loop. A stride of 16 (mod 32) puts them in
    // complementary halves and removes it entirely. Costs a few hundred bytes
    // of shared memory per channel tile.
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TH + 2;      // staged height, with the halo

    __shared__ float sh[CI][SH][SW];
    // [tap][ci][oc]: the eight `oc` weights of one (tap, ci) are contiguous, so
    // the inner loop reads them with two 16-byte loads instead of eight scalars.
    // The NG channel groups of this block sit SIDE BY SIDE along the last index,
    // group `z` owning [z*OC, z*OC + OC); with NG == 1 that is the original
    // layout exactly, so the NG == 1 instantiation is unchanged.
    __shared__ float sw[9][CI][NG * OC];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    // `tyz`/`tidz` fold the NG axis into the STAGING thread space: with NG = 2 the
    // block's 256 threads split the SAME rows between them (stride TBY * NG), so
    // the staging loop costs the block exactly what it cost at NG = 1 while
    // feeding NG times the FMAs. Without this the z-slabs would each stage the
    // whole tile - NG times the global traffic, which is the opposite of the
    // point. At NG == 1 both reduce to `ty` and `tid`.
    const int tyz = ty + (NG > 1 ? threadIdx.z : 0) * TBY;
    const int tidz = tyz * TBX + tx;
    // WHICH AXIS IS WHICH. CUDA launches blocks with `blockIdx.x` varying
    // FASTEST, and the blocks that share an input tile are exactly the ones
    // that differ only in their output-channel group. With the channels on
    // `z` (the default) those blocks are launched a whole plane apart and each
    // one re-reads the input from DRAM; with the channels on `x` they run
    // back to back and the second one's read hits L2. `ZFIRST` selects the
    // former (the layout this kernel shipped with), `CHANX` the latter.
    // THE THREE AXIS MAPS, all reachable through the same template.
    //   ZFIRST = 1                tile column on x, channel group on z (shipped)
    //   ZFIRST = 0, ZSWAP = 0     channel group on x, tile column on z
    //   ZFIRST = 0, ZSWAP = 1     BOTH on x: the tile column occupies the HIGH
    //                             bits of blockIdx.x and the channel group the
    //                             low ones, so `c_out/OC` consecutive x values
    //                             are the channel groups of ONE tile column.
    // The third is the interesting one at small plane sizes, where there are not
    // enough tiles to fill 20 SMs: it lets a kernel whose tile count is short
    // still spread its channel groups across the machine instead of running
    // `c_out/OC` of them per SM back to back. The launch must supply
    // `(GC * GW, GH, 1)` - see `gpu.rs`.
    const int GC_ = (c_out + OC - 1) / OC;
    const int bx = blockIdx.x;
    const int ox0 = (ZFIRST ? bx : (ZSWAP ? bx / GC_ : blockIdx.z)) * TW;
    const int oy0 = blockIdx.y * TH;
    // NG CHANNEL GROUPS PER BLOCK. Each group is a separate CTA elsewhere in
    // the family, so it costs its own copy of the STAGED INPUT TILE; putting
    // several groups in one block makes them share one, and that is the whole
    // point. The groups are spread over threadIdx.z (the block is
    // (TBX, TBY, NG)), and since NG does not enter the input tile at all, the
    // staging loop, its bounds predicates and the re-read are amortized over NG
    // times the arithmetic.
    //
    // WHY THIS AND NOT MORE OC. Raising `OC` cuts the re-read the same way but
    // costs acc[OC][TPX] and wv[OC] IN EVERY THREAD - `r16x`/`r32a-c` measure
    // it, and they are 9.0-11.5 ms against the 64-column tile's 6.16 at
    // 64->64@512x512. NG
    // leaves regs/thread alone: only the acc initialisation, the `wv` fetch and
    // the epilogue become NG-way, and every thread's accumulator set stays
    // `acc[OC][TPX]`.
    //
    // WHY NOT dbuf/pipe/p1. Those try to HIDE the exposed load latency and all
    // lost: a second shared buffer is a tie end to end, register prefetch is
    // -30% (the 64-column tile is at its register cap), deferred stores -22% with
    // a stack frame.
    // NG does not hide the latency, it AMORTIZES it - the same measurement that
    // condemns prefetching (72% of the kernel is the global load a shared store
    // waits on) is what endorses this one.
    //
    // MEASURED COST MODEL IT ATTACKS. One call stages GC * c_in * (TH+2)(TW+2) *
    // 4 bytes of input plus 9 * c_in * c_out * 4 * tiles of weights, the second
    // term independent of GC. At 64->64@512x512 that is 692.1 + 75.5 = 767.6 MB
    // for 19.33 GFLOP = 24.0 FLOP per staged byte, and 24.0 x the ~137 GB/s the
    // kernel actually achieves inside itself is 3288 GFLOP/s against a measured
    // 3162 - the model reproduces the ceiling, so the ceiling is the staged
    // traffic. NG=2 gives 43.7, NG=4 74.2 FLOP/byte; reaching the 8938 GFLOP/s
    // FMA ceiling at NG=1 would need 372 GB/s, above the card's DRAM peak.
    const int z = NG > 1 ? threadIdx.z : 0;
    const int oc0 = ((ZFIRST ? blockIdx.z : (ZSWAP ? bx % GC_ : bx)) * NG + z) * OC;
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;

    float acc[OC][TPX];
#pragma unroll
    for (int o = 0; o < OC; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI) {
        if (PREZERO) {
            // ONE write of the whole tile, then the staging loop only writes the
            // elements it actually has a value for. Every element of `sh` is
            // written exactly once per channel tile either way, so this is the
            // same instruction count for the tile as a whole with the bounds
            // predicates moved out of the per-element path - which is the
            // per-channel-tile overhead the staging probe showed dominates the
            // kernel (see `lg_conv3x3_stageonly`).
            float *shf = &sh[0][0][0];
            for (int i = tidz; i < CI * SH * SW; i += TBX * TBY * NG) shf[i] = 0.0f;
        }
        // Staged WITHOUT a flat index and therefore without an integer
        // division: `i / SW` is a multiply-shift sequence per staged element,
        // and there are CI of them per channel tile. The nested form walks the
        // rows and columns directly instead.
#pragma unroll
        for (int k = 0; k < CI; ++k) {
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
            for (int sy = tyz; sy < SH; sy += TBY * NG) {
                const int gy = gy0 + sy;
                const bool y_ok = ci_ok && gy >= 0 && gy < h;
                // WHICH PLANE this channel lives in, and at what offset within
                // it. With `ng == 1` the input is a single plane and this
                // reduces to the original expression. With more, the input is a
                // CONCATENATION of `ng` planes along the channel axis - the
                // dense block's growing input `cat([x, h1, h2, ...])` - and the
                // concatenated channel index is mapped to a (plane, channel
                // within plane) pair through the group starts. The comparison
                // chain runs once per staged element, its operand is the channel
                // loop index so it is uniform across a warp, and the kernel
                // reads the same bytes the concatenated copy would have held -
                // it just skips the copy (15 of them per dense block, 11% of the
                // LR64 run and 12% of the LR256 one).
                const int ci_u = ci_ok ? ci : 0;
                int base = 0;
                const float *plane = in;
                if (ng > 1 && ci_u >= g1) { base = g1; plane = i1; }
                if (ng > 2 && ci_u >= g2) { base = g2; plane = i2; }
                if (ng > 3 && ci_u >= g3) { base = g3; plane = i3; }
                if (ng > 4 && ci_u >= g4) { base = g4; plane = i4; }
                if (ng > 5 && ci_u >= g5) { base = g5; plane = i5; }
                const float *src = plane + ((size_t)(ci_u - base) * h + (gy < 0 ? 0 : gy)) * wd;
                if (PREZERO) {
                    // The whole tile was zeroed before this loop, so an element
                    // that is out of the plane is left holding that zero instead
                    // of being written with one - and the ROW predicate, which is
                    // uniform across the warp (every thread of a row shares `gy`),
                    // becomes a branch around the column loop rather than a test
                    // inside it. What remains per element is the column check
                    // alone.
                    //
                    // The row branch is also what keeps the halo at zero: rows 0
                    // and SH-1 are the halo rows and their `gy` is outside the
                    // plane for a full tile, so they take the branch and keep the
                    // zeroed value. An element at an IN-PLANE halo row (a tile
                    // that starts at row 0 has its row 0 halo row also in plane)
                    // is read back through the column check instead.
                    if (!ci_ok || gy < 0 || gy >= h) {
                        // nothing to load; the zeros stay
                    } else {
                        for (int sx = tx; sx < SW; sx += TBX) {
                            const int gx = gx0 + sx;
                            if (gx >= 0 && gx < wd) sh[k][sy][sx] = src[gx];
                        }
                    }
                } else {
                    for (int sx = tx; sx < SW; sx += TBX) {
                        const int gx = gx0 + sx;
                        float v = 0.0f;
                        if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                        sh[k][sy][sx] = v;
                    }
                }
            }
        }
        // The staged weight tile covers the NG groups' weights for THIS channel
        // tile, so its total size is 9 * CI * OC * NG - but each `z` slab owns a
        // DIFFERENT range of it (`sw[..][..][z*OC + o]`), so the split has to be
        // PER SLAB and not across the whole block: this loop's range is one
        // group's 9 * CI * OC entries and its stride is one block's worth of
        // threads. Splitting it as `tidz .. 9*CI*OC step TBX*TBY*NG` (the shape
        // the input staging correctly uses, where the staged data IS shared)
        // leaves the upper z slabs' weights unwritten - which is a WRONG PLANE,
        // not a slow one, and is exactly how this was first written and caught by
        // `--ng-test`'s NG=2 correctness row at 1.04e1 relative. The block's
        // weight-staging work therefore still grows with NG (it must - the tile
        // is NG times bigger), while the input staging does not (it must not -
        // the tile is shared).
        for (int wi = tid; wi < 9 * CI * OC; wi += TBX * TBY) {
            const int tap = wi / (CI * OC);
            const int k = (wi / OC) % CI;
            const int o = wi % OC;
            const int ci = ci0 + k;
            const int oc = oc0 + o;
            float v = 0.0f;
            if (ci < c_in && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];
            sw[tap][k][z * OC + o] = v;
        }
        __syncthreads();

        // Per (ci, tap) the TPX input values are loaded once and reused for
        // every output channel in the tile; the OC weights are loaded as OC/4
        // float4s and reused for every pixel. Raising OC raises the reuse of
        // both, at the cost of accumulators: OC * TPX floats in registers.
        //
        // With NG groups in the block, the input tile is SHARED and the weight
        // slice is not: `z` picks its group's OC weights out of the row, which is
        // still one contiguous, 16-byte-aligned run so the float4 loads below are
        // unchanged.
#pragma unroll
        for (int k = 0; k < CI; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    float v[TPX];
#pragma unroll
                    for (int p = 0; p < TPX; ++p)
                        v[p] = sh[k][ty + ky][tx + p * TBX + kx];
                    const float4 *wp =
                        reinterpret_cast<const float4 *>(&sw[tap][k][z * OC]);
                    float wv[OC];
#pragma unroll
                    for (int o4 = 0; o4 < OC / 4; ++o4) {
                        const float4 wq = wp[o4];
                        wv[o4 * 4 + 0] = wq.x;
                        wv[o4 * 4 + 1] = wq.y;
                        wv[o4 * 4 + 2] = wq.z;
                        wv[o4 * 4 + 3] = wq.w;
                    }
#pragma unroll
                    for (int o = 0; o < OC; ++o)
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv[o] * v[p];
                }
        __syncthreads();
    }

    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// grid = (ceil(wd/TW), ceil(h/TH), ceil(c_out/8)), block = (TBX, TBY, 1).
//
// The entries are written OUT rather than generated by a macro: the build script
// parses this file for `extern "C" __global__ void <name>` to check that every
// kernel it asks nvcc to keep exists (a name missing from `--entries` is PRUNED
// from the fatbin and only fails at launch), and a macro-generated name is
// invisible to that check. The signatures must stay identical to each other and
// to the toolkit's `lg_conv3x3_tile`, because `gpu.rs` launches all of them
// through one argument list.
extern "C" __global__ void __launch_bounds__(32 * 8)
lg_conv3x3_tile4(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 8, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// STAGING WITHOUT THE GLOBAL LOAD - a probe, not a candidate.
//
// `lg_conv3x3_stageonly` showed that removing 8/9 of the FMAs buys only 8-18%,
// which leaves the staging, the two barriers and the epilogue as the cost - but
// the staging is only ~400 global loads and ~400 shared stores per thread
// against 18432 FMAs, so a COUNT of instructions cannot be what makes it slow.
// The remaining explanation is the LATENCY of those loads: the shared store of a
// staged element depends on its global load, so a warp can only have as many
// loads in flight as the compiler manages to keep independent, and the kernel
// ends up limited by the number of outstanding loads rather than by DRAM
// bandwidth (which is only ~25 GB/s here, a tenth of the card's).
//
// This entry point keeps everything `lg_conv3x3_stageonly` has - the same loop
// structure, the same shared stores, the same barriers, the same epilogue - and
// replaces ONLY the value that comes from DRAM with one computed from the
// indices. It is therefore the difference between "the staging loop's
// bookkeeping and shared traffic" and "the latency of DRAM reads".
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_stage_nold(const float *__restrict__ in, const float *__restrict__ w,
                      const float *__restrict__ bias, float *__restrict__ out,
                      int c_in, int c_out, int h, int wd, int act, float act_p)
{
    constexpr int TBX = 16, TBY = 8, TPX = 4;
    constexpr int TW = TBX * TPX;
    constexpr int TH = TBY;
    constexpr int CI = 4;
    constexpr int OC = 8;
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TH + 2;
    __shared__ float sh[CI][SH][SW];
    __shared__ float sw[9][CI][OC];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.x * TW;
    const int oy0 = blockIdx.y * TH;
    const int oc0 = blockIdx.z * OC;
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;
    float acc[OC][TPX];
#pragma unroll
    for (int o = 0; o < OC; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI) {
#pragma unroll
        for (int k = 0; k < CI; ++k) {
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
            for (int sy = ty; sy < SH; sy += TBY) {
                const int gy = gy0 + sy;
                const bool y_ok = ci_ok && gy >= 0 && gy < h;
                for (int sx = tx; sx < SW; sx += TBX) {
                    const int gx = gx0 + sx;
                    // THE ONLY DIFFERENCE from `lg_conv3x3_stageonly`: the value
                    // is derived from the indices instead of loaded.
                    float v = 0.0f;
                    if (y_ok && gx >= 0 && gx < wd) v = (float)(gx + gy);
                    sh[k][sy][sx] = v;
                }
            }
        }
        if (tid < 9 * CI) {
            const int tap = tid / CI;
            const int k = tid - tap * CI;
#pragma unroll
            for (int o = 0; o < OC; ++o) sw[tap][k][o] = w[o];
        }
        __syncthreads();
#pragma unroll
        for (int o = 0; o < OC; ++o)
#pragma unroll
            for (int p = 0; p < TPX; ++p)
                acc[o][p] += sh[0][ty + 1][tx + p * TBX + 1] * sw[0][0][o];
        __syncthreads();
    }
    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// STAGING WITHOUT ARITHMETIC - a probe, not a candidate.
//
// The inner FMA loop of `lg_tile_body` is measurably not the cost: replacing
// BOTH of its shared-memory operands with constants left the time unchanged, the
// same with a 38%-FMA instruction stream instead of 26%, and every tile shape
// tried lands in the same 2.4-2.8 TFLOP/s band. That points at everything the
// loop is NOT - the staging loads, the two barriers per channel tile, the
// epilogue - so this entry point keeps all of those and drops the arithmetic,
// which is the only way to attribute the time without the hardware counters this
// user cannot read (`ncu` reports ERR_NVGPUCTRPERM).
//
// It writes a PLANE-SIZED product of the two accumulators so it cannot be
// dead-code eliminated, and it is NOT wired into the graph or the bench: it is
// reachable only as the `stage` row of `--conv-bench`, and its GFLOP/s figure is
// meaningless (it does 1/9 of the multiplies) - what matters is its ms, which is
// what `x16`'s ms has the staging cost subtracted from.
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_stageonly(const float *__restrict__ in, const float *__restrict__ w,
                     const float *__restrict__ bias, float *__restrict__ out,
                     int c_in, int c_out, int h, int wd, int act, float act_p)
{
    constexpr int TBX = 16, TBY = 8, TPX = 4;
    constexpr int TW = TBX * TPX;
    constexpr int TH = TBY;
    constexpr int CI = 4;
    constexpr int OC = 8;
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TH + 2;
    __shared__ float sh[CI][SH][SW];
    __shared__ float sw[9][CI][OC];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.x * TW;
    const int oy0 = blockIdx.y * TH;
    const int oc0 = blockIdx.z * OC;
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;
    float acc[OC][TPX];
#pragma unroll
    for (int o = 0; o < OC; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI) {
#pragma unroll
        for (int k = 0; k < CI; ++k) {
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
            for (int sy = ty; sy < SH; sy += TBY) {
                const int gy = gy0 + sy;
                const bool y_ok = ci_ok && gy >= 0 && gy < h;
                const float *src = in + ((size_t)(ci_ok ? ci : 0) * h + (gy < 0 ? 0 : gy)) * wd;
                for (int sx = tx; sx < SW; sx += TBX) {
                    const int gx = gx0 + sx;
                    float v = 0.0f;
                    if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                    sh[k][sy][sx] = v;
                }
            }
        }
        if (tid < 9 * CI) {
            const int tap = tid / CI;
            const int k = tid - tap * CI;
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
#pragma unroll
            for (int o = 0; o < OC; ++o) {
                const int oc = oc0 + o;
                float v = 0.0f;
                if (ci_ok && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];
                sw[tap][k][o] = v;
            }
        }
        __syncthreads();
        // WHERE THE ARITHMETIC WOULD BE. Reading one staged input and one staged
        // weight per output channel keeps the shared-memory traffic the same
        // shape as the real loop (it is what the smem operand count experiment
        // could not distinguish) while doing 1/9 of the multiplies.
#pragma unroll
        for (int o = 0; o < OC; ++o)
#pragma unroll
            for (int p = 0; p < TPX; ++p)
                acc[o][p] += sh[0][ty + 1][tx + p * TBX + 1] * sw[0][0][o];
        __syncthreads();
    }
    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// TW = 64 (two columns per thread) at three block heights. These exist because
// the 128-column tile is half wasted on the level-0 plains, which are 64 wide:
// the block computes 128 columns and the epilogue's bounds check discards 64 of
// them, which is a factor of two and was measured as exactly one.
extern "C" __global__ void __launch_bounds__(32 * 8)
lg_conv3x3_t2w(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 8, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(32 * 4)
lg_conv3x3_t2s(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 4, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(32 * 16)
lg_conv3x3_t2h(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 16, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// 16 output channels per tile instead of 8. The same body, with twice the
// accumulators (16 x TPX registers) and half the input staging per output, so
// each staged input value feeds 16 FMAs instead of 8. This is the shape that
// closes the gap to the LSU bound at the trunk's 64-channel layers, where the
// input staging is not amortised by a wide c_in.
extern "C" __global__ void __launch_bounds__(32 * 8)
lg_conv3x3_o16w(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 8, 2, 16>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(32 * 8)
lg_conv3x3_o16f(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 8, 4, 16>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The narrow (64-column) tile at CI = 8: four channel tiles instead of eight at
// 64 input channels, so half as much staging against the same compute. The
// staged tile is 8 * 6 * 66 * 4 = 12,672 bytes at TH = 4, which still allows
// four blocks an SM.
extern "C" __global__ void __launch_bounds__(32 * 4)
lg_conv3x3_c8s(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 4, 2, 8, 8>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The narrow tile with a 16-wide x-block: 16 * 8 threads, TW = 64 again, but
// only 64 output accumulators per WARP row, so the shared loads are spread over
// fewer threads and the staged data is reused by more of them.
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_x16(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// THE OCCUPANCY TEST THAT WAS MISSING.
//
// x16 uses 96 registers and 13,952 bytes of shared memory, so it fits three
// blocks per SM (12 warps, 41 KB). The register cap and the shared memory cap
// were never separated before: `__launch_bounds__(128, 8)` does cap registers at
// 64, but 8 x 13,952 bytes is 109 KB against the 64 KB an SM has, so occupancy
// could not rise and the experiment proved nothing.
//
// `q1` stages ONE channel at a time, which brings the tile down to ~3.5 KB and
// makes the shared memory irrelevant, and it asks for 12 blocks per SM so the
// register allocator is squeezed to 64 (the exact limit for 1536 threads).
// 12 blocks x 4 warps = 48 warps per SM against x16's 12. If the plateau is
// latency hiding - which is what is left after the instruction-count experiment
// came back null - this is the shape that moves it. CIM = 1 also means four
// times as many staging rounds, so it pays for the occupancy with staging work.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_q1(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 1>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The same 64-register request with TWO channels staged at a time (~7 KB), so
// the staging work is halved while the occupancy stays at 8 blocks / 32 warps.
// x16's tile with a pre-zeroed shared tile and no per-element row predicate.
//
// This is the tile the staging probe points at: `lg_conv3x3_stageonly` showed
// that removing 8/9 of the FMAs buys only 8-18%, so the per-channel-tile
// machinery - the staging code, the bounds predicates and the barrier pair - is
// what the kernel spends its time on. This shape keeps the staging volume and
// the tile exactly as `x16` has them and moves the work out of the per-element
// path: the tile is written once with zeros, and the staging loop then loads
// only the rows and columns that exist.
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_pz(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 4, 1, true>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// `q2` WITH THE LOADS ISSUED ONE CHANNEL TILE EARLY - a software pipeline.
//
// The staging probe chain (x16 7.05 ms, `stage` 6.19, `nold` 1.15 at
// 64->64@512x512) says the global loads of the staged input cost 5.04 ms of the
// kernel's 7.05 - and they are NOT at the DRAM limit: staging the same bytes in
// isolation, with the same grid and no arithmetic, takes 2.65 ms (692 MB at
// ~260 GB/s, essentially the card's peak), so inside the kernel the same loads
// run at half rate. What makes them slow is that they are issued and then
// immediately consumed: the staging loop does LDG, STS, LDG, STS ..., and the
// STS cannot retire until its LDG returns, so a warp has only as many loads in
// flight as the compiler dares to keep independent. The barrier after staging
// then waits for the last of them, with nothing to cover the latency. The
// staging microbenchmark does not expose this because it is all staging, all the
// time; the real kernel alternates a burst of ~20 loads with ~1150 FMAs.
//
// This kernel keeps the tile, the FMA loop, the arithmetic ORDER and the number
// of barriers exactly as `q2` has them and moves the loads one full tile
// earlier: iteration `ci0` issues the loads for tile `ci0 + CI` into registers,
// does the FMA work for tile `ci0` from shared memory, and only then writes the
// new values to shared. The load latency is therefore covered by the FMA loop
// that is already there, and the staging loop's shared stores are no longer on
// the critical path of a global load. `pre[]` costs one register per staged
// element - with CI = 2, SH = 10 rows and SW = 80 columns over 128 threads that
// is 20 - which is why the tile is `q2`'s and not `x16`'s.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_pipe(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    constexpr int TBX = 16, TBY = 8, TPX = 4;
    constexpr int TW = TBX * TPX;
    constexpr int CI = 2, OC = 8;
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TBY + 2;
    // One register per staged element: the row and column loops below both
    // stride by the thread counts, so these are the exact trip counts.
    constexpr int ROWS = (SH + TBY - 1) / TBY;
    constexpr int COLS = (SW + TBX - 1) / TBX;
    constexpr int NELEM = CI * ROWS * COLS;
    __shared__ float sh[CI][SH][SW];
    __shared__ float sw[9][CI][OC];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.x * TW;
    const int oy0 = blockIdx.y * TBY;
    const int oc0 = blockIdx.z * OC;
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;
    float acc[OC][TPX];
#pragma unroll
    for (int o = 0; o < OC; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

#pragma unroll
    for (int k = 0; k < CI; ++k) {
        const int ci = k;
        const bool ci_ok = ci < c_in;
        for (int sy = ty; sy < SH; sy += TBY) {
            const int gy = gy0 + sy;
            const bool y_ok = ci_ok && gy >= 0 && gy < h;
            const float *src = in + ((size_t)(ci_ok ? ci : 0) * h + (gy < 0 ? 0 : gy)) * wd;
            for (int sx = tx; sx < SW; sx += TBX) {
                const int gx = gx0 + sx;
                float v = 0.0f;
                if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                sh[k][sy][sx] = v;
            }
        }
    }
    if (tid < 9 * CI) {
        const int tap = tid / CI;
        const int k = tid - tap * CI;
        const bool ci_ok = k < c_in;
#pragma unroll
        for (int o = 0; o < OC; ++o) {
            const int oc = oc0 + o;
            float v = 0.0f;
            if (ci_ok && oc < c_out) v = w[((size_t)oc * c_in + k) * 9 + tap];
            sw[tap][k][o] = v;
        }
    }
    __syncthreads();

    for (int ci0 = 0; ci0 < c_in; ci0 += CI) {
        // (1) THE LOADS FOR THE NEXT TILE, issued into registers now so that the
        // FMA loop below covers their latency. The nest order is identical to
        // the store nest at (3), which is what makes the two index sequences
        // line up without carrying the coordinates through.
        const int ci1 = ci0 + CI;
        float pre[NELEM];
        {
            int np = 0;
#pragma unroll
            for (int k = 0; k < CI; ++k) {
                const int ci = ci1 + k;
                const bool ci_ok = ci < c_in;
                for (int sy = ty; sy < SH; sy += TBY) {
                    const int gy = gy0 + sy;
                    const bool y_ok = ci_ok && gy >= 0 && gy < h;
                    const float *src =
                        in + ((size_t)(ci_ok ? ci : 0) * h + (gy < 0 ? 0 : gy)) * wd;
                    for (int sx = tx; sx < SW; sx += TBX) {
                        const int gx = gx0 + sx;
                        float v = 0.0f;
                        if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                        pre[np++] = v;
                    }
                }
            }
        }
        // (2) The FMAs for THIS tile, from shared memory - identical to `q2`.
#pragma unroll
        for (int k = 0; k < CI; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    float v[TPX];
#pragma unroll
                    for (int p = 0; p < TPX; ++p)
                        v[p] = sh[k][ty + ky][tx + p * TBX + kx];
                    const float4 *wp = reinterpret_cast<const float4 *>(&sw[tap][k][0]);
                    float wv[OC];
#pragma unroll
                    for (int o4 = 0; o4 < OC / 4; ++o4) {
                        const float4 wq = wp[o4];
                        wv[o4 * 4 + 0] = wq.x;
                        wv[o4 * 4 + 1] = wq.y;
                        wv[o4 * 4 + 2] = wq.z;
                        wv[o4 * 4 + 3] = wq.w;
                    }
#pragma unroll
                    for (int o = 0; o < OC; ++o)
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv[o] * v[p];
                }
        __syncthreads();
        // (3) The values that were loaded a whole tile ago, now that no one is
        // reading the old tile. Same nest, same order, same indices.
        {
            int np = 0;
#pragma unroll
            for (int k = 0; k < CI; ++k)
                for (int sy = ty; sy < SH; sy += TBY)
                    for (int sx = tx; sx < SW; sx += TBX) sh[k][sy][sx] = pre[np++];
        }
        if (tid < 9 * CI) {
            const int tap = tid / CI;
            const int k = tid - tap * CI;
            const int ci = ci1 + k;
            const bool ci_ok = ci < c_in;
#pragma unroll
            for (int o = 0; o < OC; ++o) {
                const int oc = oc0 + o;
                float v = 0.0f;
                if (ci_ok && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];
                sw[tap][k][o] = v;
            }
        }
        __syncthreads();
    }

    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + b;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// A DOUBLE-BUFFERED TILE: the next channel tile's loads run under this one's FMAs.
//
// The probes say the 3x3 is LOAD-bound: `lg_conv3x3_stage_nold` (staging with the
// DRAM value replaced by an index-derived one) takes 64->64@512x512 from 6.19 ms
// to 1.15 ms, so the loads cost 5.04 ms of the 7.05 ms kernel while every shared
// store, both barriers and the epilogue cost 1.15 ms together. The loads are not
// at the DRAM limit either - staging the same bytes in isolation with the same
// grid is 2.65 ms (~260 GB/s, the card's peak) against 5.04 ms inside the kernel,
// so they are EXPOSED, not bandwidth-starved: the staging loop is LDG, STS, LDG,
// STS..., each STS waiting on its own load, and the barrier then waits for the
// last one with nothing to cover it.
//
// `lg_conv3x3_pipe` fixed that with registers and lost 30%, because prefetching
// 20 elements needs 20 registers and `q2` is already at its 64-register cap. This
// does the same thing WITHOUT registers: two shared buffers, the loads for tile
// ci0+CI issued into the spare buffer before the FMA loop for tile ci0, and the
// barrier at the end of the iteration covering both. The LDGs of the prefetch
// are independent of everything in the FMA loop, so the compiler can issue them
// first and let their STS land whenever the data arrives.
//
// CI = 1 rather than 2 so that doubling the tile still fits: 2 x (10 rows x 80
// columns x 4 B) + 2 weight panels = 6976 bytes, the same as `q2`'s single tile,
// so seven blocks still fit an SM (49152 / 6976 = 7). The accumulation order per
// output channel is unchanged - ci ascending, and within a channel tile the taps
// ascending - so this must agree with every other variant to the last bit.
// THE DOUBLE-BUFFERED TILE BODY. One template covers the plain conv and the
// concatenated-input conv, one channel per tile and two:
//
//   OCM  output channels per block   TPBX = TBX  (the block is TBX x TBY)
//   CIM  input channels staged per channel tile (1 or 2)
//   NG   how many source PLANES the input is a concatenation of (1 = plain)
//
// The tile is `sh[2][CIM][SH][SW]` - two buffers, so the loads for the next
// channel tile are issued into the spare one before the FMAs of the current tile
// and the single barrier per iteration covers both. That is the whole point: the
// three-probe cost model (see `lg_conv3x3_stage_nold`) says 72% of the kernel is
// the DRAM load of the staged input, and the loads are EXPOSED rather than
// bandwidth-starved (isolated staging of the same bytes with the same grid runs
// at ~260 GB/s, the card's peak, against ~137 GB/s inside the kernel). Covering
// the latency with already-existing arithmetic is free; covering it with
// registers is not (`lg_conv3x3_pipe` lost 30% because `q2` is already at its
// 64-register cap). The measured result: 64->64@512x512 4.83 ms against `q2`'s
// 6.16 and `x16`'s 6.91, and NOTHING ELSE in the tile changed.
template <int TBX, int TBY, int TPX, int OCM, int CIM, int NG, int MINB>
__device__ __forceinline__ void lg_dbuf_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p,
    const float *__restrict__ i1, const float *__restrict__ i2,
    const float *__restrict__ i3, const float *__restrict__ i4,
    const float *__restrict__ i5, int g1, int g2, int g3, int g4, int g5)
{
    constexpr int TW = TBX * TPX;
    // Conflict-free stride, computed the same way `lg_tile_body` does it: the
    // rows of the tile must not be an integer number of 32-float banks apart or
    // the column walk lands on the same bank every row.
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TBY + 2;
    __shared__ float sh[2][CIM][SH][SW];
    __shared__ float sw[2][9][CIM][OCM];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.z * TW;
    const int oy0 = blockIdx.y * TBY;
    const int oc0 = blockIdx.x * OCM;   // the channel group on blockIdx.x
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;
    float acc[OCM][TPX];
#pragma unroll
    for (int o = 0; o < OCM; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

#define LG_DBUF_STAGE(BUF, CI0)                                                \
    for (int k = 0; k < CIM; ++k) {                                            \
        const int ci = (CI0) + k;                                              \
        const bool ci_ok = ci < c_in;                                          \
        for (int sy = ty; sy < SH; sy += TBY) {                                \
            const int gy = gy0 + sy;                                           \
            const bool y_ok = ci_ok && gy >= 0 && gy < h;                      \
            const int ci_u = ci_ok ? ci : 0;                                   \
            int base = 0;                                                      \
            const float *plane = in;                                           \
            if (NG > 1 && ci_u >= g1) { base = g1; plane = i1; }               \
            if (NG > 2 && ci_u >= g2) { base = g2; plane = i2; }               \
            if (NG > 3 && ci_u >= g3) { base = g3; plane = i3; }               \
            if (NG > 4 && ci_u >= g4) { base = g4; plane = i4; }               \
            if (NG > 5 && ci_u >= g5) { base = g5; plane = i5; }               \
            const float *src =                                                 \
                plane + ((size_t)(ci_u - base) * h + (gy < 0 ? 0 : gy)) * wd;  \
            for (int sx = tx; sx < SW; sx += TBX) {                            \
                const int gx = gx0 + sx;                                       \
                float v = 0.0f;                                                \
                if (y_ok && gx >= 0 && gx < wd) v = src[gx];                   \
                sh[BUF][k][sy][sx] = v;                                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    if (tid < 9 * CIM) {                                                       \
        const int tap = tid / CIM;                                             \
        const int k = tid - tap * CIM;                                         \
        const int ci = (CI0) + k;                                              \
        const bool ci_ok = ci < c_in;                                          \
        for (int o = 0; o < OCM; ++o) {                                        \
            const int oc = oc0 + o;                                            \
            float v = 0.0f;                                                    \
            if (ci_ok && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];\
            sw[BUF][tap][k][o] = v;                                            \
        }                                                                      \
    }

    LG_DBUF_STAGE(0, 0)
    __syncthreads();

    for (int ci0 = 0; ci0 < c_in; ci0 += CIM) {
        const int b = (ci0 / CIM) & 1;
        // (1) THE NEXT TILE'S LOADS, into the spare buffer, before the FMAs that
        // will cover them. Nothing the FMA loop reads is touched.
        if (ci0 + CIM < c_in) {
            LG_DBUF_STAGE(1 - b, ci0 + CIM)
        }
        // (2) This tile's FMAs, in the same (k, ky, kx) order as `lg_tile_body`,
        // so the per-output-channel accumulation order is identical and the two
        // agree to the last bit.
#pragma unroll
        for (int k = 0; k < CIM; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    float v[TPX];
#pragma unroll
                    for (int p = 0; p < TPX; ++p)
                        v[p] = sh[b][k][ty + ky][tx + p * TBX + kx];
                    const float4 *wp =
                        reinterpret_cast<const float4 *>(&sw[b][tap][k][0]);
                    float wv[OCM];
#pragma unroll
                    for (int o4 = 0; o4 < OCM / 4; ++o4) {
                        const float4 wq = wp[o4];
                        wv[o4 * 4 + 0] = wq.x;
                        wv[o4 * 4 + 1] = wq.y;
                        wv[o4 * 4 + 2] = wq.z;
                        wv[o4 * 4 + 3] = wq.w;
                    }
#pragma unroll
                    for (int o = 0; o < OCM; ++o)
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv[o] * v[p];
                }
        // The barrier covers both sides at once: every thread has finished
        // reading buffer `b` and every thread's prefetch into `1 - b` is done.
        __syncthreads();
    }
#undef LG_DBUF_STAGE

    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OCM; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float bv = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + bv;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// The plain 3x3. Same signature as every other variant, so `gpu.rs` launches it
// through the same argument list.
extern "C" __global__ void __launch_bounds__(16 * 8, 7)
lg_conv3x3_dbuf(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_dbuf_body<16, 8, 4, 8, 1, 1, 7>(in, w, bias, out, c_in, c_out, h, wd, act,
                                       act_p, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, 0, 0, 0, 0, 0);
}

// Two input channels staged per channel tile: half the channel tiles, so half
// the barriers and half the prefetch issues, at 13952 bytes of shared memory
// (two buffers of 2 x 10 x 80 floats) against `dbuf`'s 6976.
extern "C" __global__ void __launch_bounds__(16 * 8, 7)
lg_conv3x3_dbuf2(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_dbuf_body<16, 8, 4, 8, 2, 1, 3>(in, w, bias, out, c_in, c_out, h, wd, act,
                                       act_p, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, 0, 0, 0, 0, 0);
}

// The concatenated-input conv, double buffered: the dense block's growing input
// `cat([x, h1, h2, ...])` is staged straight out of the planes that already hold
// it, which is what removes the 15 `lg_copy` launches per dense block.
extern "C" __global__ void __launch_bounds__(16 * 8, 7)
lg_conv3x3_catd(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p,
                const float *__restrict__ i1, const float *__restrict__ i2,
                const float *__restrict__ i3, const float *__restrict__ i4,
                const float *__restrict__ i5,
                int g1, int g2, int g3, int g4, int g5, int ng)
{
    (void)ng;
    lg_dbuf_body<16, 8, 4, 8, 1, 5, 7>(in, w, bias, out, c_in, c_out, h, wd, act,
                                       act_p, i1, i2, i3, i4, i5, g1, g2, g3, g4,
                                       g5);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 7)
lg_conv3x3_catd2(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p,
                 const float *__restrict__ i1, const float *__restrict__ i2,
                 const float *__restrict__ i3, const float *__restrict__ i4,
                 const float *__restrict__ i5,
                 int g1, int g2, int g3, int g4, int g5, int ng)
{
    (void)ng;
    lg_dbuf_body<16, 8, 4, 8, 2, 5, 3>(in, w, bias, out, c_in, c_out, h, wd, act,
                                       act_p, i1, i2, i3, i4, i5, g1, g2, g3, g4,
                                       g5);
}

// A REGISTER PREFETCH WITH THE SHARED STORES MOVED AFTER THE FMAS.
//
// `lg_dbuf_body` issues the next channel tile's loads into the spare shared
// buffer BEFORE the FMA loop, which puts the LDG->STS dependency stall in front
// of the arithmetic that was supposed to cover it, and therefore covers nothing.
// `lg_conv3x3_pipe` (the register version) moved the shared stores after the FMA
// loop but kept `q2`'s CIM = 2, whose 20 staged values per thread do not fit
// under `q2`'s 64-register cap, and it measured 30% slower.
//
// This is the same idea at CIM = 1, where one channel tile is 10 values per
// thread (`SH * SW / (TBX * TBY)` rounded up per row and column step) rather than
// 20, and the per-iteration order is chosen so that every global load has TWO FMA
// loops between its issue and its use:
//
//   1. store to shared the tile that was loaded one iteration ago
//   2. issue the loads for the tile two iterations ahead, into registers
//   3. the FMA loop for THIS tile, from shared memory
//   4. barrier
//
// so the load latency is covered by step 3 of this iteration AND step 3 of the
// next. Two shared buffers are enough because step 1 always writes the one step
// 3 is not reading.
//
// The point of it, from the probe chain (`lg_conv3x3_stage_nold`): 72% of the
// kernel's time is the global load of the staged input, and those loads run at
// ~137 GB/s inside the kernel against ~260 GB/s when the same bytes are staged in
// isolation with the same grid - so the total is compute + memory rather than
// max(compute, memory), which is exactly what "the loads are not covered" looks
// like.
//
// MEASURED, AND IT DOES NOT WORK. `p1` runs the trunk shape at 7.67-8.00 ms
// against `q2`'s 6.23-6.73 and `c16`'s 6.29, and nvcc reports a 40-byte stack
// frame for it: the ten staged values per thread went to LOCAL memory rather than
// registers, which is the same failure that made `lg_conv3x3_pipe` 30% slower.
// Nothing is broken and everything is in range, so `--prim-test` passes; the
// variant is kept only as the record of a third attempt at the same idea. The
// three of them together - register prefetch (-30%), shared double buffer
// (`lg_dbuf_body`, a tie end to end), register prefetch with the shared stores
// deferred (-22%) - say that the load exposure is not recoverable on this part by
// covering it, and that a kernel which reads 8x more input than it needs will pay
// for it.
template <int TBX, int TBY, int TPX, int OCM, int CIM, int NG, int MINB>
__device__ __forceinline__ void lg_pipe_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p,
    const float *__restrict__ i1, const float *__restrict__ i2,
    const float *__restrict__ i3, const float *__restrict__ i4,
    const float *__restrict__ i5, int g1, int g2, int g3, int g4, int g5)
{
    constexpr int TW = TBX * TPX;
    constexpr int SW = ((TW + 2) % 32 >= 16) ? (TW + 2)
                                             : (TW + 2 + 16 - ((TW + 2) % 32));
    constexpr int SH = TBY + 2;
    // The exact trip counts of the two staging loops below, which is what makes
    // `pre[]` indexable by a running counter: ROWS rows per thread (stride TBY)
    // and COLS columns per thread (stride TBX), for CIM channels.
    constexpr int ROWS = (SH + TBY - 1) / TBY;
    constexpr int COLS = (SW + TBX - 1) / TBX;
    constexpr int NELEM = CIM * ROWS * COLS;
    __shared__ float sh[2][CIM][SH][SW];
    __shared__ float sw[2][9][CIM][OCM];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.z * TW;
    const int oy0 = blockIdx.y * TBY;
    const int oc0 = blockIdx.x * OCM;   // the channel group on blockIdx.x
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;
    float acc[OCM][TPX];
#pragma unroll
    for (int o = 0; o < OCM; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    // Resolve a concatenated channel index to the plane that holds it and the
    // index within that plane. Uniform comparisons on a hoisted index, exactly as
    // `lg_tile_body` does it, so a caller that passes one plane and NG = 1 pays
    // nothing.
#define LG_P1_PLANE(CI_U, BASE, PLANE)                                         \
    if (NG > 1 && (CI_U) >= g1) { (BASE) = g1; (PLANE) = i1; }                 \
    if (NG > 2 && (CI_U) >= g2) { (BASE) = g2; (PLANE) = i2; }                 \
    if (NG > 3 && (CI_U) >= g3) { (BASE) = g3; (PLANE) = i3; }                 \
    if (NG > 4 && (CI_U) >= g4) { (BASE) = g4; (PLANE) = i4; }                 \
    if (NG > 5 && (CI_U) >= g5) { (BASE) = g5; (PLANE) = i5; }

    // The two staging loops walk the SAME nests in the SAME order, which is what
    // lets `pre[]` be a flat array with a running counter instead of carrying the
    // coordinates across.
#define LG_P1_LOAD(CI0)                                                        \
    {                                                                          \
        int np = 0;                                                            \
        for (int k = 0; k < CIM; ++k) {                                        \
            const int ci = (CI0) + k;                                          \
            const bool ci_ok = ci < c_in;                                      \
            const int ci_u = ci_ok ? ci : 0;                                   \
            int base = 0;                                                      \
            const float *plane = in;                                           \
            LG_P1_PLANE(ci_u, base, plane)                                     \
            for (int sy = ty; sy < SH; sy += TBY) {                            \
                const int gy = gy0 + sy;                                       \
                const bool y_ok = ci_ok && gy >= 0 && gy < h;                  \
                const float *src =                                             \
                    plane + ((size_t)(ci_u - base) * h + (gy < 0 ? 0 : gy)) * wd; \
                for (int sx = tx; sx < SW; sx += TBX) {                        \
                    const int gx = gx0 + sx;                                   \
                    float v = 0.0f;                                            \
                    if (y_ok && gx >= 0 && gx < wd) v = src[gx];               \
                    pre[np++] = v;                                             \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#define LG_P1_STORE(BUF, CI0)                                                  \
    {                                                                          \
        int np = 0;                                                            \
        for (int k = 0; k < CIM; ++k)                                          \
            for (int sy = ty; sy < SH; sy += TBY)                              \
                for (int sx = tx; sx < SW; sx += TBX) sh[BUF][k][sy][sx] = pre[np++]; \
    }
#define LG_P1_W(BUF, CI0)                                                      \
    if (tid < 9 * CIM) {                                                       \
        const int tap = tid / CIM;                                             \
        const int k = tid - tap * CIM;                                         \
        const int ci = (CI0) + k;                                              \
        const bool ci_ok = ci < c_in;                                          \
        for (int o = 0; o < OCM; ++o) {                                        \
            const int oc = oc0 + o;                                            \
            float v = 0.0f;                                                    \
            if (ci_ok && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];\
            sw[BUF][tap][k][o] = v;                                            \
        }                                                                      \
    }

    // Prologue: tile 0 by the direct route, then tile 1 into registers.
    {
        for (int k = 0; k < CIM; ++k) {
            const int ci = k;
            const bool ci_ok = ci < c_in;
            const int ci_u = ci_ok ? ci : 0;
            int base = 0;
            const float *plane = in;
            LG_P1_PLANE(ci_u, base, plane)
            for (int sy = ty; sy < SH; sy += TBY) {
                const int gy = gy0 + sy;
                const bool y_ok = ci_ok && gy >= 0 && gy < h;
                const float *src =
                    plane + ((size_t)(ci_u - base) * h + (gy < 0 ? 0 : gy)) * wd;
                for (int sx = tx; sx < SW; sx += TBX) {
                    const int gx = gx0 + sx;
                    float v = 0.0f;
                    if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                    sh[0][k][sy][sx] = v;
                }
            }
        }
        LG_P1_W(0, 0)
    }
    __syncthreads();
    float pre[NELEM];
    LG_P1_LOAD(CIM)

    for (int ci0 = 0; ci0 < c_in; ci0 += CIM) {
        const int b = (ci0 / CIM) & 1;
        // (1) The tile that was loaded one iteration ago goes to shared now.
        // (2) The loads for TWO iterations ahead are issued next, so that the FMA
        // loop below covers them and the FMA loop of the next iteration covers
        // whatever is left over.
        if (ci0 + CIM < c_in) {
            LG_P1_STORE(1 - b, ci0)
            LG_P1_W(1 - b, ci0 + CIM)
            if (ci0 + 2 * CIM < c_in) {
                LG_P1_LOAD(ci0 + 2 * CIM)
            }
        }
        // (3) This tile's FMAs. Same nest and same order as `lg_tile_body`, so
        // the per-output-channel accumulation order is identical.
#pragma unroll
        for (int k = 0; k < CIM; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky)
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    float v[TPX];
#pragma unroll
                    for (int p = 0; p < TPX; ++p)
                        v[p] = sh[b][k][ty + ky][tx + p * TBX + kx];
                    const float4 *wp =
                        reinterpret_cast<const float4 *>(&sw[b][tap][k][0]);
                    float wv[OCM];
#pragma unroll
                    for (int o4 = 0; o4 < OCM / 4; ++o4) {
                        const float4 wq = wp[o4];
                        wv[o4 * 4 + 0] = wq.x;
                        wv[o4 * 4 + 1] = wq.y;
                        wv[o4 * 4 + 2] = wq.z;
                        wv[o4 * 4 + 3] = wq.w;
                    }
#pragma unroll
                    for (int o = 0; o < OCM; ++o)
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv[o] * v[p];
                }
        // (4) The barrier protects both directions: nobody is still reading the
        // buffer just refilled, and the refill itself is complete.
        __syncthreads();
    }
#undef LG_P1_PLANE
#undef LG_P1_LOAD
#undef LG_P1_STORE
#undef LG_P1_W

    const int my = oy0 + ty;
#pragma unroll
    for (int o = 0; o < OCM; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float bv = bias ? bias[oc] : 0.0f;
#pragma unroll
            for (int p = 0; p < TPX; ++p) {
                const int gx = ox0 + tx + p * TBX;
                if (gx < wd) {
                    float v = acc[o][p] + bv;
                    if (act == 1) v = v > 0.0f ? v : 0.0f;
                    else if (act == 2) v = v >= 0.0f ? v : act_p * v;
                    out[((size_t)oc * h + my) * wd + gx] = v;
                }
            }
        }
    }
}

// The plain 3x3, and the concatenated one, both at eight blocks per SM.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_p1(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_pipe_body<16, 8, 4, 8, 1, 1, 8>(in, w, bias, out, c_in, c_out, h, wd, act,
                                       act_p, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, 0, 0, 0, 0, 0);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_catp1(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p,
                 const float *__restrict__ i1, const float *__restrict__ i2,
                 const float *__restrict__ i3, const float *__restrict__ i4,
                 const float *__restrict__ i5,
                 int g1, int g2, int g3, int g4, int g5, int ng)
{
    (void)ng;
    lg_pipe_body<16, 8, 4, 8, 1, 5, 8>(in, w, bias, out, c_in, c_out, h, wd, act,
                                       act_p, i1, i2, i3, i4, i5, g1, g2, g3, g4,
                                       g5);
}

// ---------------------------------------------------------------------------
// THE DRAM-TRAFFIC FAMILY: how many times the input plane is re-read
// ---------------------------------------------------------------------------
//
// The probes say the 3x3 is load-bound, and the amplified byte count is the
// reason: the grid is (tile columns, tile rows, c_out/OC) and every one of the
// `c_out/OC` blocks sharing a spatial tile re-reads it. At 64->64@512x512 with
// OC = 8 that is 692 MB staged for a 67 MB input, measured at 2.65 ms in
// isolation (~260 GB/s, the card's peak) and 5.04 ms inside the kernel. The only
// kernel-side lever on that number is OC, and OC is limited by the accumulator
// array: `acc[OC][TPX]` plus the weight vector `wv[OC]`. The shapes below hold
// `OC * TPX` constant at 32 (so the register pressure stays in `q2`'s band)
// while moving the product from 8x4 to 16x2 and 32x1, which cuts the re-read
// factor from 8 to 4 and 2. They are the cheap half of the family; the other
// half is a block that keeps its input tile and loops over ALL output channels,
// which needs the tile in shared memory for the whole channel extent and does
// not fit 48 KiB at these widths.
//
// All of these keep ZFIRST = 0 (the output-channel group on blockIdx.x) because
// that is what makes the blocks that share an input tile run consecutively.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_r16x(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 2, 16, 2, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_r16y(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 16, 2, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_r32a(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 4, 1, 32, 2, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_r32b(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 1, 32, 2, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_r32c(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 4, 1, 32, 4, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// A 32-COLUMN TILE FOR THE CONCATENATED CONV, for the same reason `t2s` exists:
// at 64x64 the tile count is short and the narrower tile doubles it (gw*gh = 8
// for the 64-column tile against 16 for the 64x4 one).
extern "C" __global__ void __launch_bounds__(32 * 4)
lg_conv3x3_catt2s(const float *__restrict__ in, const float *__restrict__ w,
                  const float *__restrict__ bias, float *__restrict__ out,
                  int c_in, int c_out, int h, int wd, int act, float act_p,
                  const float *__restrict__ i1, const float *__restrict__ i2,
                  const float *__restrict__ i3, const float *__restrict__ i4,
                  const float *__restrict__ i5,
                  int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<32, 4, 2, 8, 4, 1>(in, w, bias, out, c_in, c_out, h, wd, act,
                                    act_p, i1, i2, i3, i4, i5, g1, g2, g3, g4, g5,
                                    ng);
}

// THE CONCATENATED-INPUT CONV AT THE GRAPH'S REAL SMALL SHAPES.
//
// At 64x64 the dense blocks run 96->32, 128->32, 160->32 and 192->64, so the
// grid is gc = 4 or 8 output-channel groups by gh = 8 tile rows by gw = 1: 32 to
// 64 blocks in total on a 20-SM part, against a `q2` tile that wants to run
// EIGHT blocks per SM. Measured in one process, the smaller 32-column tile wins
// there by 1.5-1.8x (`t2s` 1039-1089 GFLOP/s against q2's 565-623 at
// 96->32@64x64), while at 128x128 the same comparison reverses (`q2` 2718
// against t2s 2053). Both of those kernels are the single-buffered tile with a
// different tile shape, so the missing option was one that keeps `q2`'s tile and
// registers but puts the CHANNEL GROUP AND THE TILE COLUMN BOTH ON blockIdx.x,
// which lets a grid whose tile count is short still spread every channel group
// over the machine.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_catq0s(const float *__restrict__ in, const float *__restrict__ w,
                  const float *__restrict__ bias, float *__restrict__ out,
                  int c_in, int c_out, int h, int wd, int act, float act_p,
                  const float *__restrict__ i1, const float *__restrict__ i2,
                  const float *__restrict__ i3, const float *__restrict__ i4,
                  const float *__restrict__ i5,
                  int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<16, 8, 4, 8, 2, 0, false, 1>(in, w, bias, out, c_in, c_out, h, wd,
                                              act, act_p, i1, i2, i3, i4, i5, g1,
                                              g2, g3, g4, g5, ng);
}

// The same idea for the plain conv, so `--conv-bench` can time it.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_q2s(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 2, 0, false, 1>(in, w, bias, out, c_in, c_out, h, wd,
                                              act, act_p);
}

// `q2` with the register cap of a TEN-block occupancy instead of eight.
//
// `q2`'s win over `x16` is occupancy: eight 128-thread blocks per SM instead of
// four, 32 resident warps instead of 16. The reason eight is the ceiling is that
// `__launch_bounds__(128, 8)` caps the kernel at 64 registers and 64 x 128 =
// 8192 registers is an eighth of the 65536 an SM has. Since the staging loads
// are what the kernel stalls on (see `lg_conv3x3_stageonly`: removing 8/9 of the
// FMAs buys only 8-18%), more resident warps is exactly the lever that should
// still be available - so this is the same tile under a cap of ten blocks, which
// costs registers and may spill. `--conv-bench` measures it against `q2` in one
// process rather than the argument deciding.
extern "C" __global__ void __launch_bounds__(16 * 8, 10)
lg_conv3x3_q2t(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The same small tile, but with only FOUR output channels per block so the
// accumulator file is 16 floats instead of 32. That brings the register demand
// under the 40 a 12-block request allows, so this shape gets 12 blocks x 4 warps
// = 48 warps per SM with NO spilling - the occupancy test the two variants above
// could not run. It pays for it by reusing each loaded weight across only four
// FMAs instead of eight.
extern "C" __global__ void __launch_bounds__(16 * 8, 12)
lg_conv3x3_q3(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 4, 1>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// TWO SHAPES THAT CHANGE THE INSTRUCTION-TO-FLOP RATIO, KEPT AS A NEGATIVE
// RESULT.
//
// The x16 tile's SASS is 1152 FFMA out of 5029 instructions, i.e. 4.37
// instructions per FMA, and 1152/5029 of the card's 8.9 TFLOP/s is 2.03 - the
// measured rate to three digits at 64x64. That looks like an ISSUE-BOUND
// kernel, and the lever would be instructions per FMA, not the speed of an FMA.
//
// So the tile was restructured: TPX = 8 (each thread computes EIGHT columns) so
// that one weight load and one halo load feed eight FMAs instead of four, with
// OC cut to 4 to keep the accumulator count at 32. It WORKS exactly as intended
// on paper - p8's SASS is 3174 instructions against x16's 5029, a ratio of 2.76
// instead of 4.37, with the same 1152 FMAs - and it changes the speed by
// nothing: 2560-2670 GFLOP/s against x16's 2479-2779 across 64x64, 128x128,
// 256x256, 512x512 and the 64->32 shape. The arithmetic-instruction count is
// therefore NOT the constraint; the agreement between 1152/5029 and the
// measured rate was a coincidence, and the extra integer instructions in x16
// are filling stall slots rather than blocking issue.
//
// What is left, and what p8's null result actually supports, is LATENCY: the
// inner loop issues its value loads and weight loads and consumes them within a
// few instructions, and with only 12-16 warps per SM resident there is not
// enough parallelism to cover the shared-memory latency. Halving the instruction
// count does not help a kernel that is waiting.
//
// `p8` is the 8-column/4-output tile at TW = 128; `p8n` is the same idea at
// TW = 64 / TH = 4. Both are kept because they are correct and measured.
// TWO SHAPES THAT CHANGE THE INSTRUCTION-TO-FLOP RATIO.
//
// The x16 tile's SASS is 1152 FFMA out of 5029 instructions, i.e. 4.4
// instructions per FMA, and 1152/5029 of the card's 8.9 TFLOP/s is 2.03 - the
// measured rate to three digits. So the kernel is ISSUE-BOUND at four
// instructions per SM per cycle and the lever is instructions per FMA, not the
// speed of an FMA. TPX = 8 (each thread computes EIGHT columns) halves the
// per-column loop overhead: one weight load and one halo load now feed eight
// FMAs instead of four, with OC cut to 4 so the accumulator count stays at 32.
//
// `p8` is the 8-column/4-output tile at TW = 128; `p8n` is the same at
// TW = 64 (the level-0 plane width), which keeps the block small enough to fit
// several per SM at the cost of a taller tile.
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_p8(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 8, 4, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 4)
lg_conv3x3_p8n(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 4, 4, 8, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// TWO MORE SHAPES, for the two regimes the earlier sweep left open.
//
// `w16` is the `x16` tile with a 16-row block: 16 x 16 = 256 threads, so EIGHT
// warps per block instead of four, at the SAME TW = 64 and TPX = 4. Everything
// per-thread is unchanged (same 32 accumulators, same 10 shared loads per 36
// FMAs), so if the kernel is limited by how well four warps hide shared-memory
// latency and barrier drain, eight warps at the same operand economy is exactly
// the shape that shows it.
//
// `v32` is the 128-column tile with only four rows per block: 32 x 4 = 128
// threads. It is the `tile4` geometry with the block cut in half the other way,
// which keeps TW = 128 for the wide level-1 planes without wasting the block on
// a 64-wide level-0 plane.
extern "C" __global__ void __launch_bounds__(16 * 16)
lg_conv3x3_w16(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 16, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(32 * 4)
lg_conv3x3_v32(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<32, 4, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// `x16` with the OUTPUT CHANNELS on blockIdx.x instead of blockIdx.z, so the
// blocks that share an input tile are launched consecutively and meet in L2
// instead of each re-reading the plane from DRAM. Same body, same arithmetic:
// the grid is (ceil(c_out/OC), ceil(h/TH), ceil(wd/TW)) instead of
// (ceil(wd/TW), ceil(h/TH), ceil(c_out/OC)), and `gpu::this_projects_tile`
// returns the flag so the dispatch and `--conv-bench` agree.
// THE SAME TILE WITH A CONCATENATED INPUT.
//
// The dense blocks build `cat([x, h1, h2, ...])` with device copies before every
// conv, and those copies are 11% of the LR64 run and 12% of the LR256 one (376
// ms of 3091 at LR256). The concatenated tensor is exactly the union of planes
// the engine already has, so the copies buy nothing but contiguity - and the
// staging loop below is the only consumer of that contiguity.
//
// These two entry points take up to four source planes plus the channel starts
// that define the concatenation, and are otherwise the same tile: `catq` is the
// `q2` shape (two channels staged, eight blocks per SM) and `catx` the `x16` one
// (four channels staged). The dispatch picks between them on the same block
// count `pick_by_blocks` uses.
//
// `ng` is 1 for the ordinary single-plane case, in which the staging reduces to
// the original expression; the graph only ever asks for the >1 form here.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_catq(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p,
                const float *__restrict__ i1, const float *__restrict__ i2,
                const float *__restrict__ i3, const float *__restrict__ i4,
                const float *__restrict__ i5,
                int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<16, 8, 4, 8, 2, 1>(in, w, bias, out, c_in, c_out, h, wd, act, act_p,
                                    i1, i2, i3, i4, i5, g1, g2, g3, g4, g5, ng);
}

// The same two, with the OUTPUT-CHANNEL GROUP on blockIdx.x instead of blockIdx.z.
//
// This is not a tile change at all: the arithmetic, the order of accumulation and
// the staged set are identical, so the two agree bit for bit (see the `c16` and
// `x16` entries, which differ only in this and were checked against the CPU twin
// by `--prim-test`). What changes is WHICH BLOCKS RUN NEXT TO EACH OTHER, and
// therefore how much of the input re-read the L2 still holds: with the channel
// group on `z`, the eight blocks that share an input tile are launched a whole
// plane apart, so each one re-reads the tile from DRAM. Measured at
// 64->64@512x512 in one process, `c16` (channels on x) beats `x16` (on z) by
// 8-11%, and the concatenated-input conv is 73% of the LR256 runtime - which is
// why BOTH rows of the pair carry the `0`: `catq0` is the graph's and `catx0` is
// the bench-only `x16` counterpart.
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_catx0(const float *__restrict__ in, const float *__restrict__ w,
                 const float *__restrict__ bias, float *__restrict__ out,
                 int c_in, int c_out, int h, int wd, int act, float act_p,
                 const float *__restrict__ i1, const float *__restrict__ i2,
                 const float *__restrict__ i3, const float *__restrict__ i4,
                 const float *__restrict__ i5,
                 int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<16, 8, 4, 8, 4, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p,
                                    i1, i2, i3, i4, i5, g1, g2, g3, g4, g5, ng);
}

// ---------------------------------------------------------------------------
// NG CHANNEL GROUPS PER BLOCK: why two channel groups share one staged tile
// ---------------------------------------------------------------------------
//
// THIS IS A RECORD, NOT A KERNEL: the two entry points it argues for -
// `lg_conv3x3_q2ng2` (the plain conv) and `lg_conv3x3_catq0ng2` (the
// concatenated one, which the dense blocks run) - were PROMOTED TO THE TOOLKIT
// along with the rest of the parameterised tile body, and this engine now
// dispatches the toolkit's copies by the same names. What is left here is why
// the idea works and what it measured, which is the part a reader of the
// dispatch needs; the kernel itself is in `lightgpu/cuda/kernels.cu`.
//
// THE LEVER. One call stages `GC * c_in * (TH+2)(TW+2) * 4` bytes of input, plus
// `9 * c_in * c_out * 4 * tiles` of weights, where GC = c_out/OC is the number of
// channel-group blocks per spatial tile. The weight term does not depend on GC,
// so the input term IS the amplification: at 64->64@512x512 it is 692.1 MB of
// 767.6 MB total for a 67 MB input, and the whole call is 19.33 GFLOP, i.e. 24.0
// FLOP per staged byte. 24.0 x the ~137 GB/s this kernel actually achieves
// internally is 3288 GFLOP/s against a measured 3162 - the model reproduces the
// ceiling to 4%, which is what makes it predictive rather than descriptive.
// Reaching the 8938 GFLOP/s FMA ceiling at GC = 8 would need 372 GB/s, above the
// card's 320 GB/s DRAM peak, so no scheduling change can get there.
//
// HOW. A block covers NG channel groups instead of one: the extra axis is
// threadIdx.z, the grid's channel dimension divides by NG, and the staged INPUT
// tile - the part that is read the most and costs the most to produce - is
// produced once per block and read by all NG groups. The staged WEIGHT tile
// grows by NG, but that is the 10% term.
//
//   NG   staged MB   FLOP/byte   GB/s needed at 8938 GFLOP/s
//    1      767.6        24.0     372   (above the DRAM peak: unreachable)
//    2      421.5        43.7     204
//    4      248.5        74.2     121   (at the measured in-kernel 137: FMA-bound)
//
// WHY NOT MORE OC. Raising OC divides GC too, but it costs `acc[OC][TPX]` AND
// `wv[OC]` in EVERY THREAD: `r16x`/`r32a-c` are that experiment, and they measure
// 9.0-11.5 ms against the 64-column tile's 6.16 at 64->64@512x512 because register
// pressure
// beats traffic saved. NG leaves the per-thread accumulator set at `acc[8][TPX]`;
// only the accumulator INITIALISATION, the weight-slice pointer and the epilogue
// are NG-way, and threadIdx.z is a uniform index.
//
// WHY NOT dbuf/pipe/p1. Those cover the exposed load latency: a second shared
// buffer (a tie end to end at LR256, 11% worse on this conv even in the grid
// order it wants), register prefetch (-30%, the 64-column tile is at its register
// cap), and
// prefetch with the shared stores deferred (-22%, a 40-byte stack frame because
// the array went to local memory). NG does not try to cover the latency - it
// amortizes it over NG times the arithmetic, which is why the same measurement
// that condemns prefetching (72% of the kernel is the global load a shared store
// waits on) endorses this.
//
// OCCUPANCY DOES NOT GET WORSE. NG multiplies the threads per block and the
// weight tile but leaves registers per thread at the 64-column tile's 63 and the
// input tile at
// 6976 bytes, so 65536/(63*128*NG) blocks fit by registers against 49152/6976 by
// shared memory: NG = 2 gives 4 blocks of 256 threads = 1024 threads an SM, NG = 4
// gives 2 blocks of 512 = 1024, against NG = 1's 896. The staging loop is split
// over all TBX*TBY*NG threads (stride TBY*NG), so the block's staging work is
// unchanged and the per-thread staging halves as NG doubles.
//
// THE ARITHMETIC ORDER IS UNCHANGED, per output element: the (k, ky, kx) nest
// and `acc[o][p] += wv[o] * v[p]` are untouched, and each thread still owns one
// `acc[OC][TPX]`. `--prim-test` checks the NG = 1 concatenation splits against the
// CPU twin, and `convgpu`'s dispatch can be run at NG = 2 through `--conv-bench`
// to show the two agree to the last bit.


// `q2` with the output-channel group on blockIdx.x, the `q2` counterpart of
// `c16`. Bench-only until it is measured.
extern "C" __global__ void __launch_bounds__(16 * 8, 8)
lg_conv3x3_q2x(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 2, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_catx(const float *__restrict__ in, const float *__restrict__ w,
                const float *__restrict__ bias, float *__restrict__ out,
                int c_in, int c_out, int h, int wd, int act, float act_p,
                const float *__restrict__ i1, const float *__restrict__ i2,
                const float *__restrict__ i3, const float *__restrict__ i4,
                const float *__restrict__ i5,
                int g1, int g2, int g3, int g4, int g5, int ng)
{
    lg_tile_body<16, 8, 4, 8, 4, 1>(in, w, bias, out, c_in, c_out, h, wd, act, act_p,
                                    i1, i2, i3, i4, i5, g1, g2, g3, g4, g5, ng);
}

extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_c16(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 4, 0>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The same tile with HALF the staged channels: CIM = 2 instead of 4. The block
// is still 16 x 8 threads, so this is `x16` with a smaller shared-memory
// footprint and twice as many blocks resident - one block stages 64 x 3 x 2 or
// 2 x 2 x 9 floats instead of twice that. It costs more `__syncthreads` per
// staged byte and re-reads more of the input halo, which is the trade against
// occupancy. Correctness is not in question: it is an instantiation of the same
// templated body, so it agrees with `x16` bit for bit.
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_x8(const float *__restrict__ in, const float *__restrict__ w,
              const float *__restrict__ bias, float *__restrict__ out,
              int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_tile_body<16, 8, 4, 8, 2>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// And with the channel tile between the two: CIM = 3 does not divide the 64-wide
// staged row evenly, so the staging loop rounds up and the extra columns are
// masked, which costs an integer multiply per element in the inner loop. Kept
// because the two powers of two leave a real gap - at 256 KiB of shared memory
// per SM the difference between 3 and 4 resident blocks is 33% more work in
// flight, and a shape that misses on `x16` and `x8` may land here.


// ---------------------------------------------------------------------------
// The ROW-CONTIGUOUS tile: the same algorithm with a thread's pixels made
// CONSECUTIVE instead of strided
// ---------------------------------------------------------------------------
//
// WHY. `lg_tile_body` gives each thread TPX output columns strided by TBX,
// which forces three TPX-wide scalar shared loads per (ci, ky) - one per kx -
// and TPX separate scalar stores per output channel. The kernel is therefore
// INSTRUCTION-bound, not FMA-bound: disassembled, `lg_conv3x3_x16` is 1152
// FFMA against ~2500 other instructions (674 XMAD, 495 IADD, 421 LEA, 384
// ISETP, 305 SYNC, 231 SHR, 215 IADD32I), roughly two non-FMA instructions per
// FMA. An FFMA needs an issue slot on all four schedulers to reach the
// 128 lane-ops/cycle peak, so a stream that is ~23% FMA cannot exceed ~40% of
// that peak - which is exactly where the tiled kernels sit (2740 of 8938
// measured GFLOP/s).
//
// This body produces the SAME accumulation order with fewer instructions:
//
//   * the (TPX + 2)-wide input segment a thread needs for one (ci, ky) is
//     loaded as one `float4` plus two scalars instead of three TPX-wide loads,
//     i.e. 2 shared loads where the strided body needs 12;
//   * its TPX outputs are consecutive columns, so the epilogue is ONE
//     `float4` store per output channel instead of four scalar stores with
//     four bounds tests;
//   * the per-(ci, ky, kx) address arithmetic is a shared base plus a constant
//     offset, because the thread's own column run never moves.
//
// Loads fall from 216 to ~100 per channel tile against the same 1152 FMAs and
// stores from 144 to ~36. The zero-filled halo and the (ci, ky, kx) ascending
// accumulation order are unchanged, so this agrees with `lg_tile_body` to the
// last bit and `--prim-test` covers the pair.
//
// ALIGNMENT CONTRACT: the staged row stride SW is rounded up to a multiple of
// four so every row starts on a 16-byte boundary, `sh` and `sw` are declared
// with `__align__(16)`, and a thread's column run starts at `tx * TPX` with
// TPX a multiple of four - those three together are what make the `float4`
// loads and stores below legal. The `float4` STORE additionally needs the
// output plane row stride to be a multiple of four (it is 64 or 512 in this
// network) and the tile to be a whole one (checked at run time, with a scalar
// fallback so a ragged call is still correct).
template <int TBX, int TBY, int TPX, int OCM = 8, int CIM = 4>
__device__ __forceinline__ void lg_row_body(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int c_out, int h, int wd, int act, float act_p)
{
    static_assert(TPX % 4 == 0, "the row body loads and stores 4 floats at a time");

    constexpr int TW = TBX * TPX;   // output columns per tile
    constexpr int TH = TBY;         // output rows per tile
    constexpr int CI = CIM;         // input channels per staged tile
    constexpr int OC = OCM;         // output channels per tile, accumulators per thread
    constexpr int SW = (TW + 2 + 3) / 4 * 4;   // staged width: halo, 16-byte rows
    constexpr int SH = TH + 2;                 // staged height, with the halo

    __shared__ __align__(16) float sh[CI][SH][SW];
    // [tap][ci][oc], so one (tap, ci) pair's OC weights are contiguous and the
    // inner loop loads them as OC/4 `float4`s.
    __shared__ __align__(16) float sw[9][CI][OC];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TBX + tx;
    const int ox0 = blockIdx.x * TW;
    const int oy0 = blockIdx.y * TH;
    const int oc0 = blockIdx.z * OC;
    const int gx0 = ox0 - 1, gy0 = oy0 - 1;
    const int cx = tx * TPX;        // this thread's first staged column

    float acc[OC][TPX];
#pragma unroll
    for (int o = 0; o < OC; ++o)
#pragma unroll
        for (int p = 0; p < TPX; ++p) acc[o][p] = 0.0f;

    for (int ci0 = 0; ci0 < c_in; ci0 += CI) {
        // The same nested staging walks as `lg_tile_body` - no flat index and
        // therefore no integer division per staged element.
#pragma unroll
        for (int k = 0; k < CI; ++k) {
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
            for (int sy = ty; sy < SH; sy += TBY) {
                const int gy = gy0 + sy;
                const bool y_ok = ci_ok && gy >= 0 && gy < h;
                const float *src = in + ((size_t)(ci_ok ? ci : 0) * h + (gy < 0 ? 0 : gy)) * wd;
                for (int sx = tx; sx < SW; sx += TBX) {
                    const int gx = gx0 + sx;
                    float v = 0.0f;
                    if (y_ok && gx >= 0 && gx < wd) v = src[gx];
                    sh[k][sy][sx] = v;
                }
            }
        }
        if (tid < 9 * CI) {
            const int tap = tid / CI;
            const int k = tid - tap * CI;
            const int ci = ci0 + k;
            const bool ci_ok = ci < c_in;
#pragma unroll
            for (int o = 0; o < OC; ++o) {
                const int oc = oc0 + o;
                float v = 0.0f;
                if (ci_ok && oc < c_out) v = w[((size_t)oc * c_in + ci) * 9 + tap];
                sw[tap][k][o] = v;
            }
        }
        __syncthreads();

        // Per (ci, ky) the thread's whole (TPX + 2)-wide input segment comes in
        // as one aligned float4 and two scalars, and it feeds every kx and
        // every output channel in the tile.
#pragma unroll
        for (int k = 0; k < CI; ++k)
#pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                const float *row = &sh[k][ty + ky][cx];
                float v[TPX + 2];
#pragma unroll
                for (int p = 0; p < TPX; p += 4) {
                    const float4 q = *reinterpret_cast<const float4 *>(row + p);
                    v[p + 0] = q.x;
                    v[p + 1] = q.y;
                    v[p + 2] = q.z;
                    v[p + 3] = q.w;
                }
                v[TPX + 0] = row[TPX + 0];
                v[TPX + 1] = row[TPX + 1];
#pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    const float4 *wp = reinterpret_cast<const float4 *>(&sw[tap][k][0]);
                    float wv[OC];
#pragma unroll
                    for (int o4 = 0; o4 < OC / 4; ++o4) {
                        const float4 wq = wp[o4];
                        wv[o4 * 4 + 0] = wq.x;
                        wv[o4 * 4 + 1] = wq.y;
                        wv[o4 * 4 + 2] = wq.z;
                        wv[o4 * 4 + 3] = wq.w;
                    }
#pragma unroll
                    for (int o = 0; o < OC; ++o)
#pragma unroll
                        for (int p = 0; p < TPX; ++p) acc[o][p] += wv[o] * v[p + kx];
                }
            }
        __syncthreads();
    }

    // One `float4` store per output channel when the tile is whole, which is
    // every shape this network runs (64-wide level-0 planes and 512-wide
    // level-1 ones, both multiples of TW = 64).
    const int my = oy0 + ty;
    const bool whole = (wd % 4 == 0) && (ox0 + TW <= wd);
#pragma unroll
    for (int o = 0; o < OC; ++o) {
        const int oc = oc0 + o;
        if (oc < c_out && my < h) {
            const float b = bias ? bias[oc] : 0.0f;
            float *dst = out + ((size_t)oc * h + my) * wd + ox0 + cx;
            if (whole) {
#pragma unroll
                for (int p0 = 0; p0 < TPX; p0 += 4) {
                    float4 q;
                    float t[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        float val = acc[o][p0 + p] + b;
                        if (act == 1) val = val > 0.0f ? val : 0.0f;
                        else if (act == 2) val = val >= 0.0f ? val : act_p * val;
                        t[p] = val;
                    }
                    q.x = t[0]; q.y = t[1]; q.z = t[2]; q.w = t[3];
                    *reinterpret_cast<float4 *>(dst + p0) = q;
                }
            } else {
#pragma unroll
                for (int p = 0; p < TPX; ++p) {
                    const int gx = ox0 + cx + p;
                    if (gx < wd) {
                        float val = acc[o][p] + b;
                        if (act == 1) val = val > 0.0f ? val : 0.0f;
                        else if (act == 2) val = val >= 0.0f ? val : act_p * val;
                        dst[p] = val;
                    }
                }
            }
        }
    }
}

// grid = (ceil(wd/TW), ceil(h/TH), ceil(c_out/OC)), block = (TBX, TBY, 1).
extern "C" __global__ void __launch_bounds__(16 * 8)
lg_conv3x3_r16(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_row_body<16, 8, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The same body at a 128-column tile: half as many blocks, twice as much
// reuse of every staged value. It exists because the row body's balance is
// different from the strided one's - the staging cost per FMA is what decides,
// and only a measurement says which tile wins.
extern "C" __global__ void __launch_bounds__(32 * 8)
lg_conv3x3_r32(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_row_body<32, 8, 4>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// And at 8 pixels a thread, which doubles the reuse of every weight load and
// halves the per-FMA staging again (10 shared loads per 64 FMA), paid for with
// 64 accumulators a thread.
extern "C" __global__ void __launch_bounds__(8 * 8)
lg_conv3x3_r8x(const float *__restrict__ in, const float *__restrict__ w,
               const float *__restrict__ bias, float *__restrict__ out,
               int c_in, int c_out, int h, int wd, int act, float act_p)
{
    lg_row_body<8, 8, 8>(in, w, bias, out, c_in, c_out, h, wd, act, act_p);
}

// The same idea on the row body: CIM = 2 stages 6,976 B there too, and a
// 16 x 8 block at a 64-register budget is eight blocks per SM.

// And both limits at once on the row body: a 16 x 4 block (64 threads) at
// CIM = 2 and a 32-register budget is sixteen blocks, i.e. 32 warps.


