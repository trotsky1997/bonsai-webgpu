use anyhow::{bail, Context, Result};
use half::f16;
use memmap2::Mmap;
use rand::Rng;
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

mod a4;
mod calibration;
mod chat_worker;
#[cfg(feature = "mlx")]
mod mlx_native;
mod escape_fit;
mod sidecar;
use escape_fit::escape_scales;

#[cfg(test)]
mod escape_tests;

#[cfg(target_arch = "aarch64")]
mod bitplane_neon;

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: Option<RopeScaling>,
    eos_token_id: u32,
    #[serde(default = "default_vocab")]
    vocab_size: usize,
}
#[derive(Deserialize)]
struct RopeScaling {
    factor: f32,
    original_max_position_embeddings: usize,
}
fn default_vocab() -> usize {
    151669
}

fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    candidates.truncate(top_k.max(1));
    let max = candidates[0].1;
    let mut weights: Vec<f32> = candidates
        .iter()
        .map(|(_, score)| ((score - max) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut cumulative = 0.0;
    let mut keep = weights.len();
    for (i, weight) in weights.iter().enumerate() {
        cumulative += *weight / total;
        if cumulative >= top_p.clamp(0.0, 1.0) {
            keep = i + 1;
            break;
        }
    }
    candidates.truncate(keep);
    weights.truncate(keep);
    let total: f32 = weights.iter().sum();
    let mut choice = rand::rng().random::<f32>() * total;
    for ((token, _), weight) in candidates.iter().zip(weights) {
        choice -= weight;
        if choice <= 0.0 {
            return *token as u32;
        }
    }
    candidates
        .last()
        .map(|(token, _)| *token as u32)
        .unwrap_or(0)
}

#[derive(Deserialize, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct SafeFile {
    _map: Mmap,
    base: usize,
    tensors: HashMap<String, TensorInfo>,
}
impl SafeFile {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 8 {
            bail!("safetensors file is truncated");
        }
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        if hlen > map.len() - 8 {
            bail!("invalid safetensors header length");
        }
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + hlen])?;
        let tensors = header
            .as_object()
            .context("safetensors header must be an object")?
            .iter()
            .filter(|(name, _)| name.as_str() != "__metadata__")
            .map(|(name, value)| Ok((name.clone(), serde_json::from_value(value.clone())?)))
            .collect::<Result<HashMap<String, TensorInfo>>>()?;
        Ok(Self {
            _map: map,
            base: 8 + hlen,
            tensors,
        })
    }
    fn raw(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let (a, b) = (self.base + t.data_offsets[0], self.base + t.data_offsets[1]);
        if b > self._map.len() || a > b {
            bail!("tensor {name} outside file (model file may be incomplete)");
        }
        Ok((t, &self._map[a..b]))
    }
    fn halfs(&self, name: &str) -> Result<Vec<f32>> {
        let (t, b) = self.raw(name)?;
        if t.dtype != "F16" {
            bail!("{name}: expected F16, got {}", t.dtype);
        }
        Ok(b.chunks_exact(2)
            .map(|x| f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
            .collect())
    }
    fn linear(&self, name: &str) -> Result<Linear> {
        let (wt, w) = self.raw(&format!("{name}.weight"))?;
        if wt.dtype != "U32" || wt.shape.len() != 2 {
            bail!("{name}: expected packed U32 matrix");
        }
        let out = wt.shape[0];
        let words_per_row = wt.shape[1];
        let input = words_per_row * 32;
        let scales = self.halfs(&format!("{name}.scales"))?;
        let biases = self.halfs(&format!("{name}.biases"))?;
        let groups = input / 128;
        if scales.len() != out * groups
            || biases.len() != out * groups
            || w.len() != out * words_per_row * 4
        {
            bail!("{name}: inconsistent packed tensor dimensions");
        }
        Ok(Linear {
            out,
            input,
            packed: w.to_vec(),
            binary_centered: scales.iter().zip(&biases).all(|(s, b)| *s == -2.0 * b),
            scales,
            biases,
        })
    }
}

struct Linear {
    out: usize,
    input: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    binary_centered: bool,
}

#[derive(Clone, Copy)]
struct ActivationPrecision {
    bits: usize,
    group_size: usize,
    a2_scheme: A2Scheme,
    sidecar_cache: u8,
    fast_reduction: bool,
    a4_kernel: a4::Kernel,
    a4_row_tile: usize,
}

#[derive(Clone, Copy)]
enum A2Scheme {
    Ternary,
    Oriented,
    Escape,
    Balanced,
    Hybrid,
    Sidecar(usize),
}

impl A2Scheme {
    fn name(self) -> &'static str {
        match self {
            Self::Ternary => "ternary",
            Self::Oriented => "oriented",
            Self::Escape => "escape",
            Self::Balanced => "balanced",
            Self::Hybrid => "hybrid",
            Self::Sidecar(1) => "sidecar1",
            Self::Sidecar(2) => "sidecar2",
            Self::Sidecar(4) => "sidecar4",
            Self::Sidecar(8) => "sidecar8",
            Self::Sidecar(_) => unreachable!(),
        }
    }
}

impl ActivationPrecision {
    fn intermediate(self) -> Self {
        if self.bits == 2 && matches!(self.a2_scheme, A2Scheme::Hybrid) {
            Self {
                a2_scheme: A2Scheme::Escape,
                ..self
            }
        } else {
            self
        }
    }
}

enum Activation<'a> {
    Float32 { values: &'a [f32], sums: Vec<f32> },
    Quantized(PackedActivation),
}

// Reference A4/A2 use two's-complement bitplanes. Sign-magnitude A4 uses
// [magnitude bit 0, magnitude bit 1, magnitude bit 2, negative sign].
// Layout: [activation group][bit significance][32-element word].
struct PackedActivation {
    planes: Vec<u32>,
    scales: Vec<f32>,
    sums: Vec<i32>,
    corrections: Vec<f32>,
    escape_counts: Vec<i32>,
    group_size: usize,
    bits: usize,
    fast_reduction: bool,
    a4_kernel: a4::Kernel,
    // Sign-magnitude exact: sum of negative magnitudes; fast: sum of magnitudes.
    a4_offsets: Vec<i32>,
    a4_row_tile: usize,
    balanced: bool,
    sidecar: Option<sidecar::Sidecar>,
}

impl<'a> Activation<'a> {
    fn new(values: &'a [f32], precision: ActivationPrecision) -> Self {
        if precision.bits == 32 {
            return Self::Float32 {
                values,
                sums: values.chunks_exact(128).map(|x| x.iter().sum()).collect(),
            };
        }
        if precision.bits == 4 && precision.a4_kernel != a4::Kernel::Reference {
            assert_eq!(precision.group_size, 32);
            let mut packed = a4::quantize(values, precision.a4_kernel);
            packed.a4_row_tile = precision.a4_row_tile;
            return Self::Quantized(packed);
        }
        Self::Quantized(PackedActivation::quantize(
            values,
            precision.group_size,
            precision.bits,
            if matches!(precision.a2_scheme, A2Scheme::Hybrid) {
                A2Scheme::Balanced
            } else {
                precision.a2_scheme
            },
        ))
    }
}

impl PackedActivation {
    fn quantize(values: &[f32], group_size: usize, bits: usize, a2_scheme: A2Scheme) -> Self {
        assert!(matches!(group_size, 32 | 64 | 128));
        assert_eq!(values.len() % group_size, 0);
        if bits == 2 && matches!(a2_scheme, A2Scheme::Escape) {
            return Self::quantize_escape(values, group_size);
        }
        if bits == 2 {
            if let A2Scheme::Sidecar(k) = a2_scheme {
                return sidecar::quantize(values, group_size, k);
            }
        }
        let groups = values.len() / group_size;
        let words = group_size / 32;
        let mut result = Self {
            planes: vec![0; values.len() * bits / 32],
            scales: Vec::with_capacity(groups),
            sums: Vec::with_capacity(groups),
            corrections: Vec::new(),
            escape_counts: Vec::new(),
            group_size,
            bits,
            fast_reduction: false,
            a4_kernel: a4::Kernel::Reference,
            a4_offsets: Vec::new(),
            a4_row_tile: 1,
            balanced: bits == 2 && matches!(a2_scheme, A2Scheme::Balanced),
            sidecar: None,
        };
        for (g, chunk) in values.chunks_exact(group_size).enumerate() {
            let max_abs = chunk.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            assert!(chunk.iter().all(|x| x.is_finite()), "non-finite activation");
            if bits == 2 {
                if result.balanced {
                    let (inner, outer) = balanced_scales(chunk);
                    let midpoint = (inner + outer) * 0.5;
                    for (i, &value) in chunk.iter().enumerate() {
                        let high = value >= 0.0;
                        let large = value.abs() >= midpoint;
                        let low = if high { large } else { !large };
                        let base = g * 2 * words + i / 32;
                        result.planes[base] |= u32::from(low) << (i % 32);
                        result.planes[base + words] |= u32::from(high) << (i % 32);
                    }
                    // x = c0*(2*b0-1) + c1*(2*b1-1).
                    result.scales.push((outer - inner) * 0.5);
                    result.corrections.push((outer + inner) * 0.5);
                    result.sums.push(0);
                    continue;
                }
                let (scale, escape_value) = match a2_scheme {
                    A2Scheme::Ternary => (ternary_scale(chunk), None),
                    A2Scheme::Oriented => (oriented_int2_scale(chunk), None),
                    A2Scheme::Escape => {
                        let (alpha, beta) = escape_scales(chunk);
                        (alpha, Some(beta))
                    }
                    A2Scheme::Balanced => unreachable!(),
                    A2Scheme::Hybrid | A2Scheme::Sidecar(_) => unreachable!(),
                };
                let mut sum = 0i32;
                let mut escape_count = 0i32;
                for (i, &value) in chunk.iter().enumerate() {
                    let mut q = if scale == 0.0 {
                        0i8
                    } else {
                        let lo = if matches!(a2_scheme, A2Scheme::Oriented) {
                            -2.0
                        } else {
                            -1.0
                        };
                        (value / scale).round().clamp(lo, 1.0) as i8
                    };
                    if let Some(beta) = escape_value {
                        if (value - beta).abs() < (value - scale * q as f32).abs() {
                            q = -2;
                            escape_count += 1;
                        }
                    }
                    sum += q as i32;
                    let code = (q as u8) & 3;
                    for bit in 0..2 {
                        let index = g * 2 * words + bit * words + i / 32;
                        result.planes[index] |= (((code >> bit) & 1) as u32) << (i % 32);
                    }
                }
                result.scales.push(scale);
                result.sums.push(sum);
                if let Some(beta) = escape_value {
                    result.corrections.push(beta + 2.0 * scale);
                    result.escape_counts.push(escape_count);
                }
                continue;
            }
            let scale = max_abs / 7.0;
            let mut sum = 0i32;
            for (i, &value) in chunk.iter().enumerate() {
                let q = if max_abs == 0.0 {
                    0i8
                } else {
                    ((value / max_abs) * 7.0).round().clamp(-7.0, 7.0) as i8
                };
                sum += q as i32;
                let nibble = (q as u8) & 15;
                for bit in 0..4 {
                    let index = g * 4 * words + bit * words + i / 32;
                    result.planes[index] |= (((nibble >> bit) & 1) as u32) << (i % 32);
                }
            }
            result.scales.push(scale);
            result.sums.push(sum);
        }
        result
    }

    fn quantize_escape(values: &[f32], group_size: usize) -> Self {
        let groups = values.len() / group_size;
        let words = group_size / 32;
        let mut result = Self {
            planes: vec![0; groups * words * 2],
            scales: vec![0.0; groups],
            sums: vec![0; groups],
            corrections: vec![0.0; groups],
            escape_counts: vec![0; groups],
            group_size,
            bits: 2,
            fast_reduction: false,
            a4_kernel: a4::Kernel::Reference,
            a4_offsets: Vec::new(),
            a4_row_tile: 1,
            balanced: false,
            sidecar: None,
        };
        result
            .planes
            .par_chunks_mut(2 * words)
            .zip(result.scales.par_iter_mut())
            .zip(result.sums.par_iter_mut())
            .zip(result.corrections.par_iter_mut())
            .zip(result.escape_counts.par_iter_mut())
            .zip(values.par_chunks_exact(group_size))
            .with_min_len(4)
            .for_each(
                |(((((planes, scale), sum), correction), escape_count), chunk)| {
                    assert!(chunk.iter().all(|x| x.is_finite()), "non-finite activation");
                    let (alpha, beta) = escape_scales(chunk);
                    let (mut q_sum, mut escapes) = (0i32, 0i32);
                    for (word, values) in chunk.chunks_exact(32).enumerate() {
                        let (mut low, mut high) = (0u32, 0u32);
                        for (bit, &value) in values.iter().enumerate() {
                            let mut q = if alpha == 0.0 {
                                0i8
                            } else {
                                (value / alpha).round().clamp(-1.0, 1.0) as i8
                            };
                            if (value - beta).abs() < (value - alpha * q as f32).abs() {
                                q = -2;
                                escapes += 1;
                            }
                            q_sum += q as i32;
                            low |= u32::from(q == -1 || q == 1) << bit;
                            high |= u32::from(q < 0) << bit;
                        }
                        planes[word] = low;
                        planes[words + word] = high;
                    }
                    *scale = alpha;
                    *sum = q_sum;
                    *correction = beta + 2.0 * alpha;
                    *escape_count = escapes;
                },
            );
        result
    }
}

// Exact least-squares symmetric ternary fit: alpha=mean(top-k abs(x)),
// with k maximizing sum(top-k abs(x))^2/k.
fn ternary_scale(values: &[f32]) -> f32 {
    let mut magnitudes = [0.0f32; 128];
    for (a, value) in magnitudes.iter_mut().zip(values) {
        *a = value.abs();
    }
    magnitudes[..values.len()].sort_unstable_by(|a, b| b.total_cmp(a));
    let (mut total, mut best_score, mut alpha) = (0.0f32, 0.0f32, 0.0f32);
    for (i, &value) in magnitudes[..values.len()].iter().enumerate() {
        total += value;
        let score = total * total / (i + 1) as f32;
        if score > best_score {
            best_score = score;
            alpha = total / (i + 1) as f32;
        }
    }
    alpha
}

// Exact least-squares symmetric four-level fit {-outer,-inner,+inner,+outer}.
// Sort absolute values once and enumerate the split between both magnitudes.
fn balanced_scales(values: &[f32]) -> (f32, f32) {
    let mut sorted = [0.0f64; 128];
    for (dst, &x) in sorted.iter_mut().zip(values) {
        *dst = x.abs() as f64;
    }
    let sorted = &mut sorted[..values.len()];
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let total = sorted.iter().sum::<f64>();
    let mean = total / values.len() as f64;
    let (mut best_score, mut inner, mut outer, mut sum) = (total * mean, mean, mean, 0.0);
    for k in 1..values.len() {
        sum += sorted[k - 1];
        let rest = total - sum;
        let score = sum * sum / k as f64 + rest * rest / (values.len() - k) as f64;
        if score > best_score {
            best_score = score;
            outer = sum / k as f64;
            inner = rest / (values.len() - k) as f64;
        }
    }
    (inner as f32, outer as f32)
}

// Find the global least-squares scale for q in {-2,-1,0,+1}.
// A negative scale gives the opposite orientation without another mode bit.
fn oriented_int2_scale(values: &[f32]) -> f32 {
    fn fit(values: &[f32], orientation: f64) -> (f64, f32) {
        let mut events = [(0.0f64, 0.0f64, 0.0f64); 256];
        let mut n = 0;
        let (mut numerator, mut denominator) = (0.0, 0.0);
        for &value in values {
            let u = orientation * value as f64;
            if u > 0.0 {
                numerator += u;
                denominator += 1.0;
                events[n] = (2.0 * u, u, 1.0);
                n += 1;
            } else if u < 0.0 {
                let m = -u;
                numerator += 2.0 * m;
                denominator += 4.0;
                events[n] = (2.0 * m / 3.0, m, 3.0);
                events[n + 1] = (2.0 * m, m, 1.0);
                n += 2;
            }
        }
        events[..n].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let (mut lower, mut best_scale, mut best_cost) = (0.0, 0.0, 0.0);
        for &(upper, delta_n, delta_d) in &events[..n] {
            if denominator <= 0.0 {
                break;
            }
            let scale = (numerator / denominator).clamp(lower, upper);
            let cost = scale * scale * denominator - 2.0 * scale * numerator;
            if cost < best_cost {
                best_cost = cost;
                best_scale = scale;
            }
            numerator -= delta_n;
            denominator -= delta_d;
            lower = upper;
        }
        (best_cost, (orientation * best_scale) as f32)
    }
    let negative = fit(values, 1.0);
    let positive = fit(values, -1.0);
    if negative.0 <= positive.0 {
        negative.1
    } else {
        positive.1
    }
}

impl Linear {
    fn mv(&self, input: &Activation<'_>) -> Vec<f32> {
        match input {
            Activation::Float32 { values, sums } => self.mv_f32(values, sums),
            Activation::Quantized(x) => self.mv_quantized(x),
        }
    }

    fn mv_quantized(&self, x: &PackedActivation) -> Vec<f32> {
        if x.a4_kernel != a4::Kernel::Reference {
            assert_eq!(x.bits, 4);
            assert_eq!(x.group_size, 32);
            assert_eq!(x.a4_offsets.len(), x.scales.len());
            if x.a4_kernel == a4::Kernel::SignmagFast {
                assert!(
                    self.binary_centered,
                    "signmag-fast requires centered W1 weights"
                );
            }
        }
        if x.balanced {
            assert!(
                self.binary_centered,
                "balanced A2 requires centered binary weights"
            );
        }
        #[cfg(target_arch = "aarch64")]
        if x.group_size == 32 {
            return self.mv_quantized_neon(x);
        }
        self.mv_quantized_scalar(x)
    }

    #[cfg(target_arch = "aarch64")]
    fn mv_quantized_neon(&self, x: &PackedActivation) -> Vec<f32> {
        let groups = self.input / 128;
        assert_eq!(x.planes.len(), self.input * x.bits / 32);
        assert_eq!(x.scales.len(), groups * 4);
        assert_eq!(x.sums.len(), groups * 4);
        assert_eq!(self.packed.len(), self.out * self.input / 8);
        assert_eq!(self.scales.len(), self.out * groups);
        assert_eq!(self.biases.len(), self.out * groups);
        if !x.corrections.is_empty() {
            assert_eq!(x.corrections.len(), groups * 4);
        }
        if !x.corrections.is_empty() && !x.balanced {
            assert_eq!(x.escape_counts.len(), groups * 4);
        }
        if x.a4_row_tile == 4 && x.a4_kernel == a4::Kernel::SignmagExact {
            return a4::mv_tiled_neon(self, x);
        }
        (0..self.out)
            .into_par_iter()
            .with_min_len(64)
            .map(|o| {
                // All buffers are checked above. Each row spans groups*16
                // weight bytes and groups scale/bias values. AArch64 has NEON.
                unsafe {
                    match x.a4_kernel {
                        a4::Kernel::SignmagExact => {
                            return bitplane_neon::dot_signmag_exact_g32(
                                self.packed.as_ptr().add(o * self.input / 8),
                                self.scales.as_ptr().add(o * groups),
                                self.biases.as_ptr().add(o * groups),
                                x.planes.as_ptr(),
                                x.scales.as_ptr(),
                                x.sums.as_ptr(),
                                x.a4_offsets.as_ptr(),
                                groups,
                            )
                        }
                        a4::Kernel::SignmagFast => {
                            return bitplane_neon::dot_signmag_fast_g32(
                                self.packed.as_ptr().add(o * self.input / 8),
                                self.scales.as_ptr().add(o * groups),
                                x.planes.as_ptr(),
                                x.scales.as_ptr(),
                                x.a4_offsets.as_ptr(),
                                groups,
                            )
                        }
                        a4::Kernel::Reference => {}
                    }
                    if x.bits == 4 && x.fast_reduction {
                        assert!(self.binary_centered, "fast A4 requires centered W1 weights");
                        return bitplane_neon::dot_a4_centered_g32(
                            self.packed.as_ptr().add(o * self.input / 8),
                            self.scales.as_ptr().add(o * groups),
                            x.planes.as_ptr(),
                            x.scales.as_ptr(),
                            x.sums.as_ptr(),
                            groups,
                        );
                    }
                    if let Some(side) = &x.sidecar {
                        if !side.cache.is_empty() {
                            if side.cache_mode == 2 {
                                assert!(
                                    self.binary_centered,
                                    "cached-centered requires centered W1 weights"
                                );
                                return bitplane_neon::dot_cached_centered_g32(
                                    self.packed.as_ptr().add(o * self.input / 8),
                                    self.scales.as_ptr().add(o * groups),
                                    x.planes.as_ptr(),
                                    x.scales.as_ptr(),
                                    x.sums.as_ptr(),
                                    side.cache.as_ptr(),
                                    groups,
                                );
                            }
                            return bitplane_neon::dot_cached_exact_g32(
                                self.packed.as_ptr().add(o * self.input / 8),
                                self.scales.as_ptr().add(o * groups),
                                self.biases.as_ptr().add(o * groups),
                                x.planes.as_ptr(),
                                x.scales.as_ptr(),
                                x.sums.as_ptr(),
                                side.scales.as_ptr(),
                                side.sums.as_ptr(),
                                side.cache.as_ptr(),
                                groups,
                            );
                        }
                        return bitplane_neon::dot_sidecar_g32(
                            self.packed.as_ptr().add(o * self.input / 8),
                            self.scales.as_ptr().add(o * groups),
                            self.biases.as_ptr().add(o * groups),
                            x.planes.as_ptr(),
                            x.scales.as_ptr(),
                            x.sums.as_ptr(),
                            side.scales.as_ptr(),
                            side.sums.as_ptr(),
                            side.nibbles.as_ptr(),
                            side.bytes_per_group,
                            side.per32,
                            groups,
                        );
                    }
                    if x.balanced {
                        return bitplane_neon::dot_balanced_g32(
                            self.packed.as_ptr().add(o * self.input / 8),
                            self.scales.as_ptr().add(o * groups),
                            x.planes.as_ptr(),
                            x.scales.as_ptr(),
                            x.corrections.as_ptr(),
                            groups,
                        );
                    }
                    if !x.corrections.is_empty() {
                        return bitplane_neon::dot_escape_g32(
                            self.packed.as_ptr().add(o * self.input / 8),
                            self.scales.as_ptr().add(o * groups),
                            self.biases.as_ptr().add(o * groups),
                            x.planes.as_ptr(),
                            x.scales.as_ptr(),
                            x.sums.as_ptr(),
                            x.corrections.as_ptr(),
                            x.escape_counts.as_ptr(),
                            groups,
                        );
                    }
                    if x.bits == 2 {
                        return bitplane_neon::dot_int2_g32(
                            self.packed.as_ptr().add(o * self.input / 8),
                            self.scales.as_ptr().add(o * groups),
                            self.biases.as_ptr().add(o * groups),
                            x.planes.as_ptr(),
                            x.scales.as_ptr(),
                            x.sums.as_ptr(),
                            groups,
                        );
                    }
                    bitplane_neon::dot_g32(
                        self.packed.as_ptr().add(o * self.input / 8),
                        self.scales.as_ptr().add(o * groups),
                        self.biases.as_ptr().add(o * groups),
                        x.planes.as_ptr(),
                        x.scales.as_ptr(),
                        x.sums.as_ptr(),
                        groups,
                    )
                }
            })
            .collect()
    }

    fn mv_quantized_scalar(&self, x: &PackedActivation) -> Vec<f32> {
        if x.a4_kernel != a4::Kernel::Reference {
            return a4::mv_scalar(self, x);
        }
        if x.sidecar.is_some() {
            return sidecar::mv_scalar(self, x);
        }
        debug_assert_eq!(x.planes.len() * 32 / x.bits, self.input);
        let weight_groups = self.input / 128;
        let words = x.group_size / 32;
        let groups = self.input / x.group_size;
        (0..self.out)
            .into_par_iter()
            .map(|o| {
                let row = o * self.input / 8;
                let mut sum = 0.0f32;
                if x.balanced {
                    for g in 0..groups {
                        let planes = &x.planes[g * 2 * words..(g + 1) * 2 * words];
                        let (mut d0, mut d1) = (0i32, 0i32);
                        for j in 0..words {
                            let start = row + g * x.group_size / 8 + j * 4;
                            let w = u32::from_le_bytes(
                                self.packed[start..start + 4].try_into().unwrap(),
                            );
                            d0 += 2 * (!(w ^ planes[j])).count_ones() as i32 - 32;
                            d1 += 2 * (!(w ^ planes[words + j])).count_ones() as i32 - 32;
                        }
                        let s = self.scales[o * weight_groups + g * x.group_size / 128] * 0.5;
                        sum += (s * d0 as f32) * x.scales[g] + (s * d1 as f32) * x.corrections[g];
                    }
                    return sum;
                }
                for g in 0..groups {
                    let planes = &x.planes[g * x.bits * words..(g + 1) * x.bits * words];
                    let mut dot = 0i32;
                    let mut escape_dot = 0i32;
                    for j in 0..words {
                        let start = row + g * x.group_size / 8 + j * 4;
                        let weight =
                            u32::from_le_bytes(self.packed[start..start + 4].try_into().unwrap());
                        // INT4 q = b0 + 2*b1 + 4*b2 - 8*b3.
                        // Count only positions where the packed W1 bit is one.
                        if x.bits == 2 {
                            dot += (weight & planes[j]).count_ones() as i32
                                - 2 * (weight & planes[words + j]).count_ones() as i32;
                            if !x.corrections.is_empty() {
                                escape_dot +=
                                    (weight & !planes[j] & planes[words + j]).count_ones() as i32;
                            }
                        } else {
                            dot += (weight & planes[j]).count_ones() as i32
                                + 2 * (weight & planes[words + j]).count_ones() as i32
                                + 4 * (weight & planes[2 * words + j]).count_ones() as i32
                                - 8 * (weight & planes[3 * words + j]).count_ones() as i32;
                        }
                    }
                    let wg = o * weight_groups + g * x.group_size / 128;
                    // Preserve the published MLX affine scale and bias exactly.
                    // Integer accumulation ends here; rescale each group to FP32.
                    sum += x.scales[g]
                        * (self.scales[wg] * dot as f32 + self.biases[wg] * x.sums[g] as f32);
                    if !x.corrections.is_empty() {
                        sum += x.corrections[g]
                            * (self.scales[wg] * escape_dot as f32
                                + self.biases[wg] * x.escape_counts[g] as f32);
                    }
                }
                sum
            })
            .collect()
    }

    fn mv_f32(&self, x: &[f32], x_sums: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.input);
        let groups = self.input / 128;
        (0..self.out)
            .into_par_iter()
            .map(|o| {
                let mut sum = 0.0f32;
                let row = o * self.input / 8;
                for g in 0..groups {
                    let scale = self.scales[o * groups + g];
                    let bias = self.biases[o * groups + g];
                    let start = g * 128;
                    let mut dot = 0.0f32;
                    for j in 0..16 {
                        // 8 packed bits per byte, 16 bytes per group
                        let byte = self.packed[row + g * 16 + j];
                        let x0 = start + j * 8;
                        dot += x[x0] * (byte & 1) as f32;
                        dot += x[x0 + 1] * ((byte >> 1) & 1) as f32;
                        dot += x[x0 + 2] * ((byte >> 2) & 1) as f32;
                        dot += x[x0 + 3] * ((byte >> 3) & 1) as f32;
                        dot += x[x0 + 4] * ((byte >> 4) & 1) as f32;
                        dot += x[x0 + 5] * ((byte >> 5) & 1) as f32;
                        dot += x[x0 + 6] * ((byte >> 6) & 1) as f32;
                        dot += x[x0 + 7] * ((byte >> 7) & 1) as f32;
                    }
                    // MLX affine 1-bit weights decode as bias + scale * bit, where bit is 0 or 1.
                    // Summing the common bias separately avoids materializing dequantized weights.
                    sum += scale * dot + bias * x_sums[g];
                }
                sum
            })
            .collect()
    }
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    keys: Vec<f32>,
    values: Vec<f32>,
}
struct Model {
    cfg: Config,
    tok: Tokenizer,
    embedding: Linear,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    pos: usize,
    activation_precision: ActivationPrecision,
    calibration: Option<calibration::Calibration>,
    collector: Option<calibration::Collector>,
    fingerprint: String,
}
fn rms(x: &mut [f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (ss + eps).sqrt().recip();
    for (a, b) in x.iter_mut().zip(w) {
        *a *= inv * b;
    }
}
fn rotate(v: &mut [f32], pos: usize, theta: f32, yarn: Option<&RopeScaling>) {
    let half = v.len() / 2;
    let mut freqs = Vec::with_capacity(half);
    let low_high = yarn.map(|y| {
        let corr = |rotations: f32| {
            (v.len() as f32
                * (y.original_max_position_embeddings as f32
                    / (rotations * 2.0 * std::f32::consts::PI))
                    .ln())
                / (2.0 * theta.ln())
        };
        (
            corr(32.0).floor().max(0.0).min((half - 1) as f32),
            corr(1.0).ceil().max(0.0).min((half - 1) as f32),
        )
    });
    for i in 0..half {
        let inv = theta.powf(-((2 * i) as f32) / v.len() as f32);
        let freq = if let (Some(y), Some((low, high))) = (yarn, low_high) {
            let ramp = if low == high {
                ((i as f32 - low) / 0.001).clamp(0.0, 1.0)
            } else {
                ((i as f32 - low) / (high - low)).clamp(0.0, 1.0)
            };
            let ext = 1.0 - ramp;
            inv / y.factor * (1.0 - ext) + inv * ext
        } else {
            inv
        };
        freqs.push(freq);
    }
    let attention_scale = yarn.map(|y| 1.0 + 0.1 * y.factor.ln()).unwrap_or(1.0);
    for i in 0..half {
        let angle = pos as f32 * freqs[i];
        let (s, c) = angle.sin_cos();
        let (s, c) = (s * attention_scale, c * attention_scale);
        let a = v[i];
        let b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}
impl Model {
    fn load(dir: &Path, activation_precision: ActivationPrecision) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let sf = SafeFile::open(&dir.join("model.safetensors"))?;
        let embedding = sf.linear("model.embed_tokens")?;
        if embedding.out != cfg.vocab_size || embedding.input != cfg.hidden_size {
            bail!("config/tokenizer dimensions do not match the tied embedding matrix");
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: sf.halfs(&format!("{p}.input_layernorm.weight"))?,
                post_norm: sf.halfs(&format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: sf.halfs(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: sf.halfs(&format!("{p}.self_attn.k_norm.weight"))?,
                q: sf.linear(&format!("{p}.self_attn.q_proj"))?,
                k: sf.linear(&format!("{p}.self_attn.k_proj"))?,
                v: sf.linear(&format!("{p}.self_attn.v_proj"))?,
                o: sf.linear(&format!("{p}.self_attn.o_proj"))?,
                gate: sf.linear(&format!("{p}.mlp.gate_proj"))?,
                up: sf.linear(&format!("{p}.mlp.up_proj"))?,
                down: sf.linear(&format!("{p}.mlp.down_proj"))?,
                keys: Vec::new(),
                values: Vec::new(),
            });
            if layers[i].gate.out != cfg.intermediate_size
                || layers[i].down.input != cfg.intermediate_size
                || layers[i].q.out != cfg.num_attention_heads * cfg.head_dim
                || layers[i].k.out != cfg.num_key_value_heads * cfg.head_dim
            {
                bail!("layer {i} tensor dimensions do not match config.json");
            }
        }
        if activation_precision.a4_kernel == a4::Kernel::SignmagFast
            && (!embedding.binary_centered
                || layers.iter().any(|l| {
                    [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]
                        .iter()
                        .any(|linear| !linear.binary_centered)
                }))
        {
            bail!("signmag-fast requires bias == -scale/2 in every W1 projection");
        }
        let norm = sf.halfs("model.norm.weight")?;
        Ok(Self {
            cfg,
            tok,
            embedding,
            layers,
            norm,
            pos: 0,
            activation_precision,
            calibration: None,
            collector: None,
            fingerprint: calibration::fingerprint(&sf._map),
        })
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if self.pos >= c.max_position_embeddings {
            bail!(
                "context limit of {} tokens reached",
                c.max_position_embeddings
            );
        }
        let h = c.hidden_size;
        let hd = c.head_dim;
        let nq = c.num_attention_heads;
        let nk = c.num_key_value_heads;
        let mut x = vec![0.0; h];
        // Embedding weight is stored as [vocab, hidden] and decoded row-wise.
        self.embedding.row(token as usize, &mut x)?;
        let pos = self.pos;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let mut z = x.clone();
            rms(&mut z, &layer.input_norm, c.rms_norm_eps);
            let z_input = calibration::activation(
                &z,
                self.activation_precision,
                layer_index * 4,
                self.calibration.as_ref(),
                self.collector.as_mut(),
            );
            let mut q = layer.q.mv(&z_input);
            let mut k = layer.k.mv(&z_input);
            let v = layer.v.mv(&z_input);
            for head in 0..nq {
                let a = &mut q[head * hd..(head + 1) * hd];
                rms(a, &layer.q_norm, c.rms_norm_eps);
                rotate(a, pos, c.rope_theta, c.rope_scaling.as_ref());
            }
            for head in 0..nk {
                let a = &mut k[head * hd..(head + 1) * hd];
                rms(a, &layer.k_norm, c.rms_norm_eps);
                rotate(a, pos, c.rope_theta, c.rope_scaling.as_ref());
            }
            layer.keys.extend_from_slice(&k);
            layer.values.extend_from_slice(&v);
            let mut att = vec![0.0; h];
            let scale = (hd as f32).sqrt().recip();
            let seq = pos + 1;
            for qh in 0..nq {
                let kh = qh / (nq / nk);
                let qv = &q[qh * hd..(qh + 1) * hd];
                let mut scores = Vec::with_capacity(seq);
                for t in 0..seq {
                    let key = &layer.keys[t * nk * hd + kh * hd..t * nk * hd + (kh + 1) * hd];
                    scores.push(qv.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0.0;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    den += *s;
                }
                for (t, s) in scores.iter().enumerate() {
                    let weight = *s / den;
                    let val = &layer.values[t * nk * hd + kh * hd..t * nk * hd + (kh + 1) * hd];
                    for d in 0..hd {
                        att[qh * hd + d] += weight * val[d];
                    }
                }
            }
            let o = layer.o.mv(&calibration::activation(
                &att,
                self.activation_precision.intermediate(),
                layer_index * 4 + 1,
                self.calibration.as_ref(),
                self.collector.as_mut(),
            ));
            for i in 0..h {
                x[i] += o[i];
            }
            let mut z = x.clone();
            rms(&mut z, &layer.post_norm, c.rms_norm_eps);
            let z_input = calibration::activation(
                &z,
                self.activation_precision,
                layer_index * 4 + 2,
                self.calibration.as_ref(),
                self.collector.as_mut(),
            );
            let mut gate = layer.gate.mv(&z_input);
            let up = layer.up.mv(&z_input);
            for (a, b) in gate.iter_mut().zip(up) {
                *a = (*a / (1.0 + (-*a).exp())) * b;
            }
            let down = layer.down.mv(&calibration::activation(
                &gate,
                self.activation_precision.intermediate(),
                layer_index * 4 + 3,
                self.calibration.as_ref(),
                self.collector.as_mut(),
            ));
            for i in 0..h {
                x[i] += down[i];
            }
        }
        self.pos += 1;
        rms(&mut x, &self.norm, c.rms_norm_eps);
        Ok(self.embedding.mv(&calibration::activation(
            &x,
            self.activation_precision,
            self.layers.len() * 4,
            self.calibration.as_ref(),
            self.collector.as_mut(),
        )))
    }
    fn reset_cache(&mut self) {
        self.pos = 0;
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }
    fn prompt_logits(
        &mut self,
        prompt: &str,
        add_special_tokens: bool,
    ) -> Result<(Vec<f32>, usize, f64)> {
        self.reset_cache();
        let ids = self
            .tok
            .encode(prompt, add_special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.is_empty() {
            bail!("tokenizer returned an empty prompt");
        }
        let start = Instant::now();
        let mut logits = Vec::new();
        for id in &ids {
            logits = self.forward(*id)?;
        }
        Ok((logits, ids.len(), start.elapsed().as_secs_f64()))
    }
    fn answer(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        timings: bool,
        ignore_eos: bool,
    ) -> Result<String> {
        let (mut logits, prompt_tokens, prefill_s) = self.prompt_logits(prompt, true)?;
        let mut generated = Vec::new();
        let decode_start = Instant::now();
        let mut decode_steps = 0usize;
        for _ in 0..max_tokens {
            let id = sample(&logits, temperature, top_k, top_p);
            if id == self.cfg.eos_token_id && !ignore_eos {
                break;
            }
            generated.push(id);
            if generated.len() < max_tokens {
                logits = self.forward(id)?;
                decode_steps += 1;
            }
        }
        let decode_s = decode_start.elapsed().as_secs_f64();
        if timings {
            eprintln!(
                "TIMING prompt_tokens={} prefill_s={:.6} prefill_tok_s={:.2} generated_tokens={} decode_steps={} decode_s={:.6} decode_tok_s={:.2} activation_bits={} activation_group_size={} a2_scheme={} a4_kernel={} a4_row_tile={}",
                prompt_tokens,
                prefill_s,
                prompt_tokens as f64 / prefill_s,
                generated.len(),
                decode_steps,
                decode_s,
                if decode_steps > 0 { decode_steps as f64 / decode_s } else { 0.0 },
                self.activation_precision.bits,
                if self.activation_precision.bits != 32 { self.activation_precision.group_size } else { 0 },
                self.activation_precision.a2_scheme.name(),
                self.activation_precision.a4_kernel.name(),
                self.activation_precision.a4_row_tile,
            );
            eprintln!("GENERATION_IDS={}", serde_json::to_string(&generated)?);
        }
        self.tok
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}
impl Linear {
    fn row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        if row >= self.out || out.len() != self.input {
            bail!("embedding index/dim mismatch");
        }
        let groups = self.input / 128;
        let base = row * self.input / 8;
        for i in 0..self.input {
            let b = self.packed[base + i / 8];
            let bit = (b >> (i % 8)) & 1;
            let g = i / 128;
            out[i] = self.biases[row * groups + g] + self.scales[row * groups + g] * bit as f32;
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct MmluCase {
    subject: String,
    prompt: String,
    answer: usize,
    shots: usize,
    sample_seed: u64,
}

fn eval_mmlu(model: &mut Model, input: &Path, report: &Path) -> Result<()> {
    let file = File::open(input).with_context(|| format!("open {}", input.display()))?;
    let mut predictions = Vec::new();
    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_prefill_tokens = 0usize;
    let mut total_prefill_s = 0.0f64;
    let mut candidate_ids = None;
    let mut shots = None;
    let mut sample_seed = None;
    for (line_no, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: MmluCase = serde_json::from_str(&line)
            .with_context(|| format!("parse MMLU row {}", line_no + 1))?;
        if shots.is_some_and(|value| value != case.shots)
            || sample_seed.is_some_and(|value| value != case.sample_seed)
        {
            bail!("MMLU rows must use consistent shots and sample_seed metadata");
        }
        shots = Some(case.shots);
        sample_seed = Some(case.sample_seed);
        if case.answer >= 4 {
            bail!(
                "MMLU answer index must be between 0 and 3 at row {}",
                line_no + 1
            );
        }
        let ctx = model
            .tok
            .encode(case.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        let mut answer_tokens = Vec::with_capacity(4);
        for label in ["A", "B", "C", "D"] {
            let full = model
                .tok
                .encode(format!("{} {}", case.prompt, label), false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();
            if !full.starts_with(&ctx) || full.len() != ctx.len() + 1 {
                bail!("MMLU continuation '{label}' is not one token after the prompt; cannot score exactly");
            }
            answer_tokens.push(full[ctx.len()]);
        }
        if let Some(expected) = &candidate_ids {
            if expected != &answer_tokens {
                bail!("tokenizer candidate IDs changed between rows");
            }
        } else {
            candidate_ids = Some(answer_tokens.clone());
        }
        model.reset_cache();
        let start = Instant::now();
        let mut logits = Vec::new();
        for id in &ctx {
            logits = model.forward(*id)?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        total_prefill_tokens += ctx.len();
        total_prefill_s += elapsed;
        let scores: Vec<f32> = answer_tokens
            .iter()
            .map(|id| logits[*id as usize])
            .collect();
        let predicted = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let correct = predicted == case.answer;
        let labels = ["A", "B", "C", "D"];
        let gold_label = labels[case.answer];
        let predicted_label = labels[predicted];
        let entry = counts.entry(case.subject.clone()).or_insert((0, 0));
        entry.0 += usize::from(correct);
        entry.1 += 1;
        predictions.push(serde_json::json!({
            "subject": case.subject,
            "gold": gold_label,
            "prediction": predicted_label,
            "scores": scores,
            "correct": correct,
            "prompt_tokens": ctx.len(),
            "prefill_seconds": elapsed,
        }));
        if predictions.len() % 10 == 0 {
            eprintln!("MMLU progress: {} questions", predictions.len());
        }
    }
    let correct: usize = counts.values().map(|x| x.0).sum();
    let total: usize = counts.values().map(|x| x.1).sum();
    let per_subject: HashMap<String, serde_json::Value> = counts
        .into_iter()
        .map(|(subject, (n_correct, n))| {
            (
                subject,
                serde_json::json!({ "correct": n_correct, "total": n, "accuracy": n_correct as f64 / n as f64 }),
            )
        })
        .collect();
    let result = serde_json::json!({
        "benchmark": "MMLU",
        "dataset": "cais/mmlu test",
        "few_shot_examples": shots,
        "sample_seed": sample_seed,
        "scoring": "single-token conditional logit for A/B/C/D; accuracy",
        "questions": total,
        "subjects": per_subject.len(),
        "correct": correct,
        "accuracy": correct as f64 / total as f64,
        "mean_prefill_tokens_per_second": total_prefill_tokens as f64 / total_prefill_s,
        "total_prompt_tokens": total_prefill_tokens,
        "total_prefill_seconds": total_prefill_s,
        "per_subject": per_subject,
        "predictions": predictions,
    });
    std::fs::write(report, serde_json::to_vec_pretty(&result)?)
        .with_context(|| format!("write {}", report.display()))?;
    println!(
        "MMLU: {correct}/{total} = {:.2}% across {} subjects; prompt prefill {:.2} tok/s; report: {}",
        100.0 * correct as f64 / total as f64,
        per_subject.len(),
        total_prefill_tokens as f64 / total_prefill_s,
        report.display()
    );
    Ok(())
}

#[derive(Deserialize)]
struct DiagnosticText {
    name: String,
    text: String,
}

fn score_diagnostics(model: &mut Model, input: &Path, output: &Path) -> Result<()> {
    let mut cases = Vec::new();
    let mut nll_sum = 0.0f64;
    let mut scored_tokens = 0usize;
    for line in io::BufReader::new(File::open(input)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: DiagnosticText = serde_json::from_str(&line)?;
        let ids = model
            .tok
            .encode(case.text.as_str(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.len() < 2 {
            bail!("diagnostic text {} needs at least two tokens", case.name);
        }
        model.reset_cache();
        let mut nll = Vec::with_capacity(ids.len() - 1);
        let mut predicted = Vec::with_capacity(ids.len() - 1);
        for pair in ids.windows(2) {
            let logits = model.forward(pair[0])?;
            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            let max = *best.1 as f64;
            let normalizer = logits
                .iter()
                .map(|x| (*x as f64 - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            nll.push(normalizer - logits[pair[1] as usize] as f64);
            predicted.push(best.0 as u32);
        }
        let total = nll.iter().sum::<f64>();
        nll_sum += total;
        scored_tokens += nll.len();
        eprintln!(
            "DIAGNOSTIC name={} tokens={} mean_nll={:.4}",
            case.name,
            nll.len(),
            total / nll.len() as f64
        );
        cases.push(serde_json::json!({
            "name": case.name, "text": case.text, "token_ids": ids,
            "scored_tokens": nll.len(), "mean_nll": total / nll.len() as f64,
            "next_token_nll": nll, "predicted_token_ids": predicted,
        }));
    }
    if scored_tokens == 0 {
        bail!("diagnostic corpus is empty");
    }
    let mean = nll_sum / scored_tokens as f64;
    let report = serde_json::json!({
        "kind": "fixed authored text diagnostic, teacher forcing; not a standard benchmark",
        "activation_bits": model.activation_precision.bits,
        "activation_group_size": model.activation_precision.group_size,
        "a4_kernel": model.activation_precision.a4_kernel.name(),
        "a4_row_tile": model.activation_precision.a4_row_tile,
        "a2_scheme": model.activation_precision.a2_scheme.name(),
        "scored_tokens": scored_tokens, "mean_nll": mean, "diagnostic_perplexity": mean.exp(),
        "cases": cases,
    });
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "tokens={scored_tokens} mean_nll={mean:.4} diagnostic_perplexity={:.3}",
        mean.exp()
    );
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--chat-worker") {
        return chat_worker::run();
    }
    let dedicated_a4 = env!("CARGO_BIN_NAME") == "bonsai-w1a4-symmetric";
    let mut args = std::env::args_os().skip(1).peekable();
    if args.peek().is_some_and(|x| x == "--a4-microbench") {
        return a4::microbench();
    }
    if args.peek().is_some_and(|x| x == "--help" || x == "-h") {
        println!("{} MODEL_DIR [--prompt TEXT] [--max-tokens N] [--temperature T] [--top-k N] [--top-p P] [--timings] [--ignore-eos] [--activation-bits 2|4|32] [--a2-scheme ternary|oriented|escape|balanced|hybrid|sidecar1|sidecar2|sidecar4|sidecar8] [--activation-group-size 32|64|128] [--score-jsonl FILE --score-report FILE] [--calibrate-jsonl FILE --calibration-report FILE] [--sidecar-calibration FILE] [--sidecar-kernel compact|cached-exact|cached-centered] [--fast-reduction] [--a4-kernel reference|signmag-exact|signmag-fast] [--a4-row-tile 1|4]\nDedicated bonsai-w1a4-symmetric defaults to W1A4 g32 signmag-exact, row tile 1. signmag-fast is experimental (quality gate failed).\nDefault for bonsai-w1a2: W1A2 escape, four-level activations (-alpha,0,+alpha,beta) in groups of 32, two packed bitplanes, INT32 dot products, FP32 group rescaling. RAYON_NUM_THREADS controls CPU threads (default min(CPUs,12)).", env!("CARGO_BIN_NAME"));
        return Ok(());
    }
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "model".into()));
    let mut max_tokens = 128usize;
    let mut temperature = 0.5f32;
    let mut top_k = 20usize;
    let mut top_p = 0.9f32;
    let mut question = String::new();
    let mut timings = false;
    let mut ignore_eos = false;
    let mut activation_precision = ActivationPrecision {
        bits: if dedicated_a4 { 4 } else { 2 },
        group_size: 32,
        a2_scheme: A2Scheme::Escape,
        sidecar_cache: 0,
        fast_reduction: false,
        a4_kernel: if dedicated_a4 {
            a4::Kernel::SignmagExact
        } else {
            a4::Kernel::Reference
        },
        a4_row_tile: 1,
    };
    let mut explicit_a4_kernel = false;
    let mut explicit_a4_row_tile = false;
    let mut calibration_input: Option<PathBuf> = None;
    let mut calibration_output = PathBuf::from("sidecar-calibration.json");
    let mut calibration_file: Option<PathBuf> = None;
    let mut mmlu_input: Option<PathBuf> = None;
    let mut mmlu_report = PathBuf::from("mmlu-report.json");
    let mut diagnostic_input: Option<PathBuf> = None;
    let mut diagnostic_report = PathBuf::from("diagnostic-report.json");
    while let Some(a) = args.next() {
        match a.to_string_lossy().as_ref() {
            "--a4-kernel" => {
                activation_precision.a4_kernel = match args
                    .next()
                    .context("--a4-kernel needs reference, signmag-exact, or signmag-fast")?
                    .to_string_lossy()
                    .as_ref()
                {
                    "reference" => a4::Kernel::Reference,
                    "signmag-exact" => a4::Kernel::SignmagExact,
                    "signmag-fast" => a4::Kernel::SignmagFast,
                    _ => bail!("invalid --a4-kernel"),
                };
                explicit_a4_kernel = true;
            }
            "--a4-row-tile" => {
                explicit_a4_row_tile = true;
                activation_precision.a4_row_tile = args
                    .next()
                    .context("--a4-row-tile needs 1 or 4")?
                    .to_string_lossy()
                    .parse()?;
                if !matches!(activation_precision.a4_row_tile, 1 | 4) {
                    bail!("--a4-row-tile must be 1 or 4");
                }
            }
            "--fast-reduction" => activation_precision.fast_reduction = true,
            "--calibrate-jsonl" => {
                calibration_input = Some(PathBuf::from(
                    args.next().context("--calibrate-jsonl needs a path")?,
                ))
            }
            "--calibration-report" => {
                calibration_output =
                    PathBuf::from(args.next().context("--calibration-report needs a path")?)
            }
            "--sidecar-calibration" => {
                calibration_file = Some(PathBuf::from(
                    args.next().context("--sidecar-calibration needs a path")?,
                ))
            }
            "--sidecar-kernel" => {
                activation_precision.sidecar_cache = match args
                    .next()
                    .context("--sidecar-kernel needs compact, cached-exact, or cached-centered")?
                    .to_string_lossy()
                    .as_ref()
                {
                    "compact" => 0,
                    "cached-exact" => 1,
                    "cached-centered" => 2,
                    _ => bail!("invalid sidecar kernel"),
                }
            }
            "--score-jsonl" => {
                diagnostic_input = Some(PathBuf::from(
                    args.next().context("--score-jsonl needs a path")?,
                ));
            }
            "--score-report" => {
                diagnostic_report =
                    PathBuf::from(args.next().context("--score-report needs a path")?);
            }
            "--activation-bits" => {
                activation_precision.bits = args
                    .next()
                    .context("--activation-bits needs 2, 4 or 32")?
                    .to_string_lossy()
                    .parse()?;
                if !matches!(activation_precision.bits, 2 | 4 | 32) {
                    bail!("--activation-bits must be 2, 4 or 32");
                }
            }
            "--a2-scheme" => {
                activation_precision.a2_scheme = match args
                    .next()
                    .context("--a2-scheme needs a scheme name (see --help)")?
                    .to_string_lossy()
                    .as_ref()
                {
                    "ternary" => A2Scheme::Ternary,
                    "oriented" => A2Scheme::Oriented,
                    "escape" => A2Scheme::Escape,
                    "balanced" => A2Scheme::Balanced,
                    "hybrid" => A2Scheme::Hybrid,
                    "sidecar1" => A2Scheme::Sidecar(1),
                    "sidecar2" => A2Scheme::Sidecar(2),
                    "sidecar4" => A2Scheme::Sidecar(4),
                    "sidecar8" => A2Scheme::Sidecar(8),
                    _ => bail!("unknown --a2-scheme (see --help)"),
                };
            }
            "--activation-group-size" => {
                activation_precision.group_size = args
                    .next()
                    .context("--activation-group-size needs 32, 64, or 128")?
                    .to_string_lossy()
                    .parse()?;
                if !matches!(activation_precision.group_size, 32 | 64 | 128) {
                    bail!("--activation-group-size must be 32, 64, or 128");
                }
            }
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .context("--max-tokens needs a value")?
                    .to_string_lossy()
                    .parse()?
            }
            "--temperature" => {
                temperature = args
                    .next()
                    .context("--temperature needs a value")?
                    .to_string_lossy()
                    .parse()?;
            }
            "--top-k" => {
                top_k = args
                    .next()
                    .context("--top-k needs a value")?
                    .to_string_lossy()
                    .parse()?;
            }
            "--top-p" => {
                top_p = args
                    .next()
                    .context("--top-p needs a value")?
                    .to_string_lossy()
                    .parse()?;
            }
            "--timings" => timings = true,
            "--ignore-eos" => ignore_eos = true,
            "--mmlu-jsonl" => {
                mmlu_input = Some(PathBuf::from(
                    args.next().context("--mmlu-jsonl needs a path")?,
                ));
            }
            "--mmlu-report" => {
                mmlu_report = PathBuf::from(args.next().context("--mmlu-report needs a path")?);
            }
            "--prompt" => {
                question = args
                    .next()
                    .context("--prompt needs text")?
                    .to_string_lossy()
                    .into_owned()
            }
            x => bail!("unknown argument {x}"),
        }
    }
    if activation_precision.bits != 4 {
        if explicit_a4_kernel {
            bail!("--a4-kernel requires --activation-bits 4");
        }
        activation_precision.a4_kernel = a4::Kernel::Reference;
    }
    if activation_precision.a4_kernel != a4::Kernel::Reference {
        if activation_precision.group_size != 32 {
            bail!("sign-magnitude A4 kernels require --activation-group-size 32");
        }
        if activation_precision.fast_reduction {
            bail!("--fast-reduction is a legacy kernel; use --a4-kernel signmag-fast instead");
        }
    }
    if !explicit_a4_row_tile && activation_precision.a4_kernel != a4::Kernel::SignmagExact {
        activation_precision.a4_row_tile = 1;
    }
    if activation_precision.a4_row_tile == 4
        && activation_precision.a4_kernel != a4::Kernel::SignmagExact
    {
        bail!("--a4-row-tile 4 requires --activation-bits 4 --a4-kernel signmag-exact");
    }
    if calibration_input.is_some()
        || calibration_file.is_some()
        || activation_precision.sidecar_cache > 0
    {
        if activation_precision.bits != 2
            || activation_precision.group_size != 32
            || !matches!(activation_precision.a2_scheme, A2Scheme::Sidecar(2))
        {
            bail!("calibration/cache requires --activation-bits 2 --a2-scheme sidecar2 --activation-group-size 32");
        }
    }
    if calibration_input.is_some() && calibration_file.is_some() {
        bail!("collect calibration without loading a fitted calibration");
    }
    if question.is_empty()
        && mmlu_input.is_none()
        && diagnostic_input.is_none()
        && calibration_input.is_none()
    {
        print!("User: ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut question)?;
    }
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(12)
        });
    if threads == 0 {
        bail!("RAYON_NUM_THREADS must be positive");
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    // Keep the caller inside the pool across the entire autoregressive loop.
    // Each projection then uses local Rayon joins instead of an external
    // submission and condition-variable wakeup for every individual matvec.
    pool.install(|| {
        let mut model = Model::load(&dir, activation_precision)?;
        if let Some(file) = calibration_file {
            let calibration: calibration::Calibration =
                serde_json::from_slice(&std::fs::read(file)?)?;
            calibration.validate(&model)?;
            model.calibration = Some(calibration);
        }
        if let Some(input) = calibration_input {
            return calibration::collect(&mut model, &input, &calibration_output);
        }
        if let Some(input) = diagnostic_input {
            return score_diagnostics(&mut model, &input, &diagnostic_report);
        }
        if let Some(input) = mmlu_input {
            return eval_mmlu(&mut model, &input, &mmlu_report);
        }
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n",
            question.trim()
        );
        let answer = model.answer(
            &prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            timings,
            ignore_eos,
        )?;
        println!("{answer}");
        Ok(())
    })
}
