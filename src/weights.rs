//! Weights: the HCFlow reading of a converted `.safetensors` checkpoint.
//!
//! The container itself - `mmap`, the JSON header, offsets and borrowed slices -
//! is `lightgpu::safetensors`, shared with every other engine in the family.
//! What is left here is what is specific to HCFlow:
//!
//! * the architecture constants are read out of `__metadata__`;
//! * every tensor the graph walks is shape-checked against those constants at
//!   load, because a mis-converted checkpoint should fail here and not produce a
//!   plausible-looking image;
//! * `get()` returns a zero-copy `&[f32]` into the mapping.
//!
//! THE LEVEL NUMBERING IS THE ENGINE'S, NOT THE CHECKPOINT'S.  HCFlow's
//! inference walks the reference's layer list BACKWARDS, so it runs the
//! 24-channel block first and the 12-channel block second; the checkpoint calls
//! the 24-channel one "level1" (`flow.layers.16..28` plus `flow.level1_condFlow`).
//! This module names levels the way the *walk* visits them, and the converter
//! writes them in that same order, so `l0` is the 24-channel level and
//! `for level in 0..2` is the order the graph runs in.
//!
//! THE COUPLING'S WIDTH IS NOT THE STEP'S WIDTH.  A conditional step's `z` is
//! `cond_in` channels wide, which is typically ODD (21 at level 0).  The
//! reference's `split_feature(h, "split")` then gives halves of `cond_in / 2`
//! (floor) and `cond_in - cond_in / 2`, and its conv3 emits `2 *
//! (cond_in - cond_in / 2)` channels, i.e. one shift/scale pair per channel of
//! the SECOND half.  Every width in this file comes from that rule.
use lightgpu::safetensors::File;

/// The architecture constants the engine needs before it can walk the graph.
#[derive(Clone, Debug)]
pub struct Config {
    pub scale: usize,
    pub n_levels: usize,
    /// Channels of `z` entering each level (the width the Split consumes).
    pub in_ch: Vec<usize>,
    /// Channels of `z` the Split cat adds - also the conditional flow's `f`
    /// input width and the conditional steps' `z` width.
    pub c_split: Vec<usize>,
    /// Channels of `z` the level's unconditional steps run at
    /// (`in_ch + c_split`, as measured from the checkpoint).
    pub full_ch: Vec<usize>,
    /// Channels of the conditional steps' `z`.
    pub cond_in: Vec<usize>,
    /// Channels of the conditional flow's `f` output (`2 * cond_in`).
    pub cond_out: Vec<usize>,
    /// Channels of the cached conditional feature.
    pub cond_ch: Vec<usize>,
    /// How many levels of conditional feature feed each level's `conv_first`.
    pub cond_levels: Vec<usize>,
    /// Unconditional and conditional flow steps per level.
    pub steps: Vec<usize>,
    pub cond_steps: Vec<usize>,
    /// Number of `RRDB` blocks per trunk, read from the file's tensor names.
    pub rrdb_blocks: usize,
    /// The reference's `eps_std`, recorded so a run can say which temperature it
    /// is sampling at even though the caller supplies the actual epsilon.
    pub eps_std: f32,
    /// The coupling nets' hidden width, which the checkpoint does not name:
    /// every `conv1`/`conv2` weight's output channel count is this.
    pub hidden: usize,
    /// The dense blocks' growth channel, likewise unnamed in the file.
    pub gc: usize,
}

impl Config {
    /// The coupling's first half: `n / 2`, floor.  This is the reference's
    /// `split_feature(..., "split")` rule.
    pub fn half(n: usize) -> usize {
        n / 2
    }

    /// The coupling's second half, `n - n / 2` (one more than `half` when `n`
    /// is odd).
    pub fn half2(n: usize) -> usize {
        n - n / 2
    }
}

pub struct Weights {
    pub file: File,
    pub config: Config,
}

/// Tensor names are built by these helpers so a typo is a compile-time-ish
/// error in one place rather than a runtime miss in thirty.
pub fn step(base: &str, i: usize) -> String {
    format!("{base}.steps.{i}")
}

pub fn cond_step(base: &str, i: usize) -> String {
    format!("{base}.cond_steps.{i}")
}

/// One dense block of a trunk: `{base}.blocks.{blk}.{rdb}`.
pub fn rdb(base: &str, blk: usize, rdb: usize) -> String {
    format!("{base}.blocks.{blk}.{rdb}")
}

impl Weights {
    pub fn open(path: &str) -> Result<Weights, String> {
        let file = File::open(path).map_err(|e| format!("{e} (not a converted HCFlow file?)"))?;
        let get = |k: &str| -> Result<usize, String> {
            file.metadata_usize(k).map_err(|_| {
                format!("checkpoint metadata is missing `{k}` (not a converted HCFlow file?)")
            })
        };
        let get_list = |k: &str| -> Result<Vec<usize>, String> {
            let s = file
                .metadata_get(k)
                .ok_or_else(|| format!("checkpoint metadata is missing `{k}`"))?;
            s.split(',')
                .map(|v| v.trim().parse::<usize>().map_err(|e| format!("`{k}` = {s:?}: {e}")))
                .collect()
        };
        let config = Config {
            scale: get("scale")?,
            n_levels: get("n_levels")?,
            in_ch: get_list("in_ch")?,
            c_split: get_list("c_split")?,
            full_ch: get_list("full_ch")?,
            cond_in: get_list("cond_in")?,
            cond_out: get_list("cond_out")?,
            cond_ch: get_list("cond_ch")?,
            cond_levels: get_list("cond_levels")?,
            steps: get_list("steps")?,
            cond_steps: get_list("cond_steps")?,
            rrdb_blocks: 0, // filled by check_shapes, which counts them
            eps_std: file
                .metadata_get("eps_std")
                .unwrap_or("0.9")
                .parse::<f32>()
                .map_err(|e| format!("eps_std: {e}"))?,
            hidden: 64,
            gc: 32,
        };
        let mut w = Weights { file, config };
        w.check_shapes()?;
        Ok(w)
    }

    pub fn get(&self, name: &str) -> Result<&[f32], String> {
        self.file.f32(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        self.file.shape(name)
    }

    /// Sum of the tensor payloads, for the load message.
    pub fn total_bytes(&self) -> usize {
        self.file.payload_bytes()
    }

    pub fn num_tensors(&self) -> usize {
        self.file.order().len()
    }

    /// How many `RRDB` blocks a trunk has, counted from the file's own names.
    /// The config's `RRDB_nb` is not trusted anywhere in this engine.
    fn count_blocks(&self, level: usize, trunk: usize) -> usize {
        let base = format!("l{level}.trunk{trunk}.blocks.");
        let mut n = 0;
        while self.file.shape(&format!("{base}{n}.0.w1")).is_ok() {
            n += 1;
        }
        n
    }

    /// Check the shapes against the architecture the metadata claims. A
    /// mismatch here means a corrupted or wrongly-converted checkpoint, and
    /// catching it at load is far cheaper than debugging a wrong image.
    fn check_shapes(&mut self) -> Result<(), String> {
        let mut found_blocks = 0usize;
        let c = &self.config;
        if c.n_levels != 2 {
            return Err(format!("this engine walks 2 levels, the file has {}", c.n_levels));
        }
        if c.scale != 4 {
            return Err(format!("this engine is the x4 model, the file is x{}", c.scale));
        }
        for v in [&c.in_ch, &c.c_split, &c.full_ch, &c.cond_in, &c.cond_out,
                  &c.cond_ch, &c.cond_levels, &c.steps, &c.cond_steps] {
            if v.len() != c.n_levels {
                return Err(format!("metadata list length {} != n_levels {}", v.len(), c.n_levels));
            }
        }

        let expect = |name: &str, want: &[usize]| -> Result<(), String> {
            let got = self.shape(name)?;
            if got != want {
                return Err(format!("{}: shape {:?}, expected {:?}", name, got, want));
            }
            Ok(())
        };

        for level in 0..c.n_levels {
            let b = format!("l{level}");
            let ic = c.in_ch[level];
            let cs = c.c_split[level];
            let full = c.full_ch[level];
            let ci = c.cond_in[level];
            let co = c.cond_out[level];
            let cond_ch = c.cond_ch[level];
            // `full_ch` is the width the UNCONDITIONAL steps run at, which is
            // what the level holds after the Split inverse cats the sampled
            // latent on: `in_ch + cond_in`. `c_split` is the width the
            // conditional flow samples, and at inference time that is the whole
            // of `z` as it enters - the Split that consumes it comes later in
            // the walk than the conditional flow does.
            if full != ic + ci {
                return Err(format!(
                    "level {level}: full_ch {full} != in_ch {ic} + cond_in {ci}"
                ));
            }
            if cs != ic {
                return Err(format!(
                    "level {level}: c_split {cs} != in_ch {ic} (the Split consumes all of z)"
                ));
            }
            if co != 2 * ci {
                return Err(format!("level {level}: cond_out {co} != 2 * cond_in {ci}"));
            }

            // The unconditional steps run at the full (post-concat) width.
            let nsteps = c.steps[level];
            for i in 0..nsteps {
                let s = step(&b, i);
                expect(&format!("{s}.an_scale"), &[full])?;
                expect(&format!("{s}.an_bias"), &[full])?;
                expect(&format!("{s}.w"), &[full, full])?;
                expect(&format!("{s}.w_rev"), &[full, full])?;
                expect(&format!("{s}.c1w"), &[c.hidden, Config::half(full), 3, 3])?;
                expect(&format!("{s}.c1b"), &[c.hidden])?;
                // The 1x1 keeps torch's 4D shape ([out][in][1][1]).
                expect(&format!("{s}.c2w"), &[c.hidden, c.hidden, 1, 1])?;
                expect(&format!("{s}.c2b"), &[c.hidden])?;
                expect(&format!("{s}.c3w"), &[2 * Config::half2(full), c.hidden, 3, 3])?;
                expect(&format!("{s}.c3b"), &[2 * Config::half2(full)])?;
            }

            // The conditional steps run at `cond_in`, with the 128-channel
            // feature appended to the coupling's first half.
            for i in 0..c.cond_steps[level] {
                let s = cond_step(&b, i);
                expect(&format!("{s}.an_scale"), &[ci])?;
                expect(&format!("{s}.an_bias"), &[ci])?;
                expect(&format!("{s}.w"), &[ci, ci])?;
                expect(&format!("{s}.w_rev"), &[ci, ci])?;
                expect(&format!("{s}.c1w"), &[c.hidden, Config::half(ci) + cond_ch, 3, 3])?;
                expect(&format!("{s}.c1b"), &[c.hidden])?;
                expect(&format!("{s}.c2w"), &[c.hidden, c.hidden, 1, 1])?;
                expect(&format!("{s}.c2b"), &[c.hidden])?;
                expect(&format!("{s}.c3w"), &[2 * Config::half2(ci), c.hidden, 3, 3])?;
                expect(&format!("{s}.c3b"), &[2 * Config::half2(ci)])?;
            }

            // The conditional feature builder: conv_first mixes the level's own
            // `in_ch` with (cond_levels * cond_ch) cached feature channels.
            let feat_in = ic + c.cond_levels[level] * cond_ch;
            expect(&format!("{b}.cond_conv_first_w"), &[c.hidden, feat_in, 3, 3])?;
            expect(&format!("{b}.cond_conv_first_b"), &[c.hidden])?;

            let blocks = self.count_blocks(level, 0);
            let blocks1 = self.count_blocks(level, 1);
            if blocks == 0 || blocks1 != blocks {
                return Err(format!("level {level}: trunks have {blocks} / {blocks1} blocks"));
            }
            found_blocks = found_blocks.max(blocks);
            for trunk in 0..2 {
                for blk in 0..blocks {
                    for r in 0..3 {
                        let base = rdb(&format!("{b}.trunk{trunk}"), blk, r);
                        // Dense block: conv1 reads nf, each later conv reads nf
                        // plus every previous growth output.
                        expect(&format!("{base}.w1"), &[c.gc, c.hidden, 3, 3])?;
                        expect(&format!("{base}.b1"), &[c.gc])?;
                        expect(&format!("{base}.w2"), &[c.gc, c.hidden + c.gc, 3, 3])?;
                        expect(&format!("{base}.b2"), &[c.gc])?;
                        expect(&format!("{base}.w3"), &[c.gc, c.hidden + 2 * c.gc, 3, 3])?;
                        expect(&format!("{base}.b3"), &[c.gc])?;
                        expect(&format!("{base}.w4"), &[c.gc, c.hidden + 3 * c.gc, 3, 3])?;
                        expect(&format!("{base}.b4"), &[c.gc])?;
                        expect(&format!("{base}.w5"), &[c.hidden, c.hidden + 4 * c.gc, 3, 3])?;
                        expect(&format!("{base}.b5"), &[c.hidden])?;
                    }
                }
            }
            expect(&format!("{b}.trunk_conv_w"), &[c.hidden, c.hidden, 3, 3])?;
            expect(&format!("{b}.trunk_conv_b"), &[c.hidden])?;
            // `f` reads the 128-channel feature, not a trunk's 64.
            expect(&format!("{b}.cond_out_w"), &[co, c.cond_ch[level], 3, 3])?;
            expect(&format!("{b}.cond_out_b"), &[co])?;
        }
        self.config.rrdb_blocks = found_blocks;
        Ok(())
    }
}
