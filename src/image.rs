//! PNG IO: decode to NCHW f32 in [0,1], encode RGB output.
//!
//! The reference scales by 1/255 into fp32, runs the network, clamps to [0,1]
//! and rounds back to 8-bit. This module does exactly that, on the host side of
//! the boundary, and it is the only conversion path: the device-side kernels
//! that once duplicated it were removed as unexercised.
use std::fs::File;
use std::io::{BufWriter, Read, Write};

/// A planar RGB image: `data` is [3][h][w], contiguous, values in [0,1].
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize) -> Image {
        Image { w, h, data: vec![0.0; 3 * w * h] }
    }

    /// The image as interleaved RGB bytes, clamped and rounded the way the
    /// reference rounds (`(x * 255).round()`).
    pub fn to_rgb8(&self) -> Vec<u8> {
        let hw = self.w * self.h;
        let mut out = vec![0u8; hw * 3];
        for i in 0..hw {
            for c in 0..3 {
                let v = self.data[c * hw + i].clamp(0.0, 1.0) * 255.0;
                out[i * 3 + c] = (v + 0.5).floor().min(255.0) as u8;
            }
        }
        out
    }
}

/// Decode a PNG (8/16-bit, gray/RGB/RGBA) into NCHW f32 RGB in [0,1].
pub fn load_rgb(path: &str) -> Result<Image, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
    load_rgb_stream(file).map_err(|e| format!("{}: {}", path, e))
}

/// The same decode, from any reader - used for `-i -`, where the PNG arrives on
/// stdin. The whole stream is buffered because the decoder needs the header
/// before it can size the frame.
pub fn load_rgb_stream<R: Read>(src: R) -> Result<Image, String> {
    let decoder = png::Decoder::new(src);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("unsupported png colour type {:?}", other)),
    };
    let sixteen = info.bit_depth == png::BitDepth::Sixteen;
    let sample_scale = match info.bit_depth {
        png::BitDepth::Eight => 1.0 / 255.0,
        png::BitDepth::Sixteen => 1.0 / 65535.0,
        other => return Err(format!("unsupported png bit depth {:?}", other)),
    };
    let bytes = &buf[..info.buffer_size()];
    let esize = if sixteen { 2 } else { 1 };
    let sample = |px: usize, c: usize| -> f32 {
        let idx = px * channels + c;
        if sixteen {
            let b0 = bytes[idx * 2] as u16;
            let b1 = bytes[idx * 2 + 1] as u16;
            (((b0 << 8) | b1) as f32) * sample_scale
        } else {
            bytes[idx] as f32 * sample_scale
        }
    };
    let mut img = Image::new(w, h);
    let hw = w * h;
    let _ = esize;
    for y in 0..h {
        for x in 0..w {
            let px = y * w + x;
            let (r, g, b) = match channels {
                1 | 2 => {
                    let v = sample(px, 0);
                    (v, v, v)
                }
                _ => (sample(px, 0), sample(px, 1), sample(px, 2)),
            };
            img.data[px] = r;
            img.data[hw + px] = g;
            img.data[2 * hw + px] = b;
        }
    }
    Ok(img)
}

/// Write 8-bit RGB.
pub fn save_rgb(path: &str, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("create {}: {}", path, e))?;
    save_rgb_stream(BufWriter::new(file), w, h, rgb)
        .map_err(|e| format!("{}: {}", path, e))?;
    Ok(())
}

/// The same encode, to any writer - used for `-o -`, where the PNG goes to
/// stdout. The writer is flushed here so the bytes are out before the process
/// exits.
pub fn save_rgb_stream<W: Write>(dst: W, w: usize, h: usize, rgb: &[u8]) -> Result<(), String> {
    assert_eq!(rgb.len(), w * h * 3);
    let mut enc = png::Encoder::new(dst, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgb).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(())
}
