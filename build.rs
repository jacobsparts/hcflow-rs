//! Compiles this engine's kernel list: the shared `lightgpu` toolkit's
//! `cuda/kernels.cu` (TOOLKIT_KERNELS), plus this project's own
//! `cuda/hcflow.cu` (PROJECT_KERNELS) for the ops the toolkit has no
//! equivalent for.
//!
//! Each source compiles to its own fatbin with its own `--entries` list and
//! `src/cuda.rs` loads them as separate modules, so neither can shadow a name
//! in the other. A kernel missing from its list is PRUNED from the fatbin and
//! fails at launch rather than at build time, so both lists are checked against
//! the source they are compiled from before nvcc runs.
//!
//! The convolutions, activations, per-channel affines and all THREE of the
//! permutations - pixel-unshuffle by 2, nearest 2x upsample, and the
//! pixel-shuffle that inverts the first - come from the toolkit.
//! `cuda/hcflow.cu` adds only `lg_couple_inverse`, which fuses the coupling's
//! split-into-shift-and-scale, its `0.318 * atan(2 * scale)` and the inversion
//! of the lower half of `z` into one pass - as separate kernels it would be
//! three extra planes per coupling and there are 52 couplings per level - and
//! `lg_sample_latent`, the conditional flow's latent sampler.

/// Generic ops from the shared toolkit.
const TOOLKIT_KERNELS: &[&str] = &[
    // Convolutions. 3x3 s1 p1 with a nullable bias FIRST in the accumulator
    // (the same order the CPU twin uses, so the two backends can be compared
    // with a tolerance rather than a mood); 1x1 as a plain GEMM-per-pixel.
    //
    // These two are the hot kernels by a wide margin: the conditional feature
    // is two RRDB trunks of seven blocks of three dense blocks of five 3x3
    // convs, i.e. 210 3x3 convs per level, against 52 coupling convs. The
    // tiled and Winograd 3x3 variants in the toolkit are the obvious
    // alternatives to measure against `lg_conv3x3s1p1` once the graph is
    // numerically right.
    "lg_conv3x3s1p1",
    "lg_conv1x1",
    // The register-blocked 1x1: a 64x64-pixel by 16-channel GEMM tile in shared
    // memory instead of one thread per output element. `lg_conv1x1` walks c_in
    // with a stride of `h*w` for EVERY output, so it re-reads the whole input
    // plane once per output channel and reads it uncoalesced - measured at ~0.5
    // TFLOP/s on the trunk shapes, 591 ms of a 3944 ms LR256 run for 70 GFLOP.
    "lg_conv1x1_rb",
    // The TILED 3x3, staged in shared memory: 16x the throughput of the
    // untiled one at 64->64 (the toolkit's own measurement) and the same
    // signature apart from a trailing `act`/`act_p`. Its accumulation order is
    // (ci, ky, kx) rather than the untiled kernel's (ky, kx, ci), so the two are
    // not bit-identical; `gpu.rs` keeps both and `--prim-test` checks whichever
    // one the dispatch runs against the CPU twin.
    "lg_conv3x3_tile",
    // THE PARAMETERISED TILE BODY, which was THIS ENGINE's and now lives in the
    // toolkit. `q2` is the tile the dev campaign's sweep selected and what the
    // graph dispatches; `q2ng2` is the same tile with two output-channel groups
    // per block; `catq0`/`catq0ng2` are the concatenated-input forms the dense
    // blocks run. Same accumulation order as `lg_conv3x3_tile` (ci, ky, kx, bias
    // after the sum), so all four are BIT-IDENTICAL to the copies that were here
    // - which is what the promote-then-compare sequence proved against the
    // pre-promotion binary before the project's copies were deleted.
    "lg_conv3x3_q2",
    "lg_conv3x3_q2ng2",
    "lg_conv3x3_catq0",
    "lg_conv3x3_catq0ng2",
    // The toolkit's F(4x4,3x3) Winograd: 4/9ths of the multiplies of the direct
    // form, at the cost of an input transform, a weight transform and an inverse
    // transform. It wins only when the channel count is large enough to amortise
    // those, which is exactly this network's trunk (64 -> 64 at 3x3). The
    // PROJECT's own winograd was measured and removed (1.4-2.0 TFLOP/s); this is
    // the toolkit's, and `--conv-bench` is how it would be measured again.
    "lg_conv3x3_winograd",
    // Per-channel affine: the folded ActNorms and the coupling's
    // exp(-logscale) scale. `lg_channel_scale` is the shift-free case.
    "lg_channel_affine",
    // Elementwise. `lg_relu` and `lg_lrelu` are the two activations the
    // coupling and the RRDB blocks use; `lg_add_scaled` is the residual
    // `a + s * b` the RRDB block ends with, and `lg_scale`/`lg_copy` build the
    // concatenations without a host round trip (`lg_copy` doubles as the
    // device-side copy that `cat` needs).
    "lg_relu",
    "lg_lrelu",
    "lg_add_scaled",
    "lg_scale",
    "lg_copy",
    // All three permutations: pixel-unshuffle by 2 (the SqueezeLayer), nearest 2x
    // upsampling (the level-0 conditional feature), and the pixel-shuffle that
    // inverts the first (the Unsqueeze2d that ends the network, at r = 2 it is
    // exactly that inverse - see `gpu::unsqueeze2d` for the measurement that put
    // it here).
    "lg_pixel_unshuffle2",
    "lg_upsample2x_nearest",
    "lg_pixel_shuffle",
];

/// This project's own kernels - the ops the toolkit has no equivalent for.
///
/// `lg_couple_inverse` is the coupling's fused inversion: it takes the network's
/// output plane, splits even/odd channels into shift and scale, applies
/// `logscale = 0.318 * atan(2 * scale)` and inverts the second half of `z` in one
/// pass. Doing that on the host would be three round-trips per coupling, and the
/// per-pixel `atan`/`exp` in a plan of its own would cost another two planes of
/// 2 * half channels each. `lg_sample_latent` is the conditional flow's latent
/// sampler, which needs the whole level's statistics in one place.
///
/// The Unsqueeze2d inverse used to be the third entry here. It is the toolkit's
/// `lg_pixel_shuffle` at r = 2 - the two were measured against each other in one
/// process and the toolkit's form won everywhere that has work to do - so this
/// list no longer has it and `gpu::unsqueeze2d` dispatches the toolkit's.
const PROJECT_KERNELS: &[&str] = &[
    "lg_couple_inverse",
    "lg_sample_latent",
    // The engine's own tiled 3x3: the toolkit's with CI = 4 instead of 8 (so
    // two blocks fit per SM) and a transposed weight panel (so the inner loop
    // reads its output-channel weights as float4s). Same algebra, same
    // accumulation order, same halo - see cuda/hcflow.cu.
    "lg_conv3x3_tile4",
    // The same body at a 64-column tile, in three block shapes, so the tile
    // shape can be measured rather than guessed at on this engine's 64-wide
    // level-0 planes.
    "lg_conv3x3_t2w",
    "lg_conv3x3_t2s",
    "lg_conv3x3_t2h",
    // And two at 16 output channels per tile, where the register pressure allows.
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit"
        );
    }
    if !PROJECT_KERNELS.is_empty() {
        println!("cargo:rerun-if-changed=cuda/hcflow.cu");
        let src = std::fs::read_to_string("cuda/hcflow.cu").expect("read cuda/hcflow.cu");
        let defined = lightgpu_build::kernel_names_in(&src);
        for k in PROJECT_KERNELS {
            assert!(
                defined.iter().any(|d| d == k),
                "`{k}` is not defined in cuda/hcflow.cu (it has {})",
                defined.join(", ")
            );
        }
    }

    let mut sources = vec![lightgpu_build::Source {
        path: &toolkit,
        out_name: "hcflow_toolkit.fatbin",
        entries: Some(TOOLKIT_KERNELS),
    }];
    if !PROJECT_KERNELS.is_empty() {
        sources.push(lightgpu_build::Source {
            path: "cuda/hcflow.cu",
            out_name: "hcflow_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        });
    }
    lightgpu_build::fatbin_modules(&sources);
}
