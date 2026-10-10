//! Exact [-7,7] A4 values, encoded as sign plus three magnitude bitplanes.
use crate::{Linear, PackedActivation};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    Reference,
    SignmagExact,
    SignmagFast,
}

impl Kernel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::SignmagExact => "signmag-exact",
            Self::SignmagFast => "signmag-fast",
        }
    }
}

pub(crate) fn quantize(values: &[f32], kernel: Kernel) -> PackedActivation {
    assert_ne!(kernel, Kernel::Reference);
    assert_eq!(values.len() % 32, 0);
    let groups = values.len() / 32;
    let mut result = PackedActivation {
        planes: vec![0; groups * 4],
        scales: Vec::with_capacity(groups),
        sums: Vec::with_capacity(groups),
        corrections: Vec::new(),
        escape_counts: Vec::new(),
        group_size: 32,
        bits: 4,
        fast_reduction: false,
        a4_kernel: kernel,
        a4_offsets: Vec::with_capacity(groups),
        a4_row_tile: 1,
        balanced: false,
        sidecar: None,
    };
    for (g, chunk) in values.chunks_exact(32).enumerate() {
        let max_abs = chunk.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        assert!(chunk.iter().all(|x| x.is_finite()), "non-finite activation");
        let mut planes = [0u32; 4];
        let (mut sum, mut absolute) = (0i32, 0i32);
        for (i, &value) in chunk.iter().enumerate() {
            // Keep the exact FP32 quantizer expression used by the reference.
            let q = if max_abs == 0.0 {
                0i8
            } else {
                ((value / max_abs) * 7.0).round().clamp(-7.0, 7.0) as i8
            };
            let magnitude = q.unsigned_abs();
            sum += q as i32;
            absolute += magnitude as i32;
            planes[0] |= ((magnitude & 1) as u32) << i;
            planes[1] |= (((magnitude >> 1) & 1) as u32) << i;
            planes[2] |= (((magnitude >> 2) & 1) as u32) << i;
            planes[3] |= u32::from(q < 0) << i;
        }
        result.planes[g * 4..g * 4 + 4].copy_from_slice(&planes);
        result.scales.push(max_abs / 7.0);
        result.sums.push(sum);
        result.a4_offsets.push(if kernel == Kernel::SignmagExact {
            (absolute - sum) / 2 // Sum of negative magnitudes.
        } else {
            absolute
        });
    }
    result
}

pub(crate) fn mv_scalar(linear: &Linear, x: &PackedActivation) -> Vec<f32> {
    let groups = linear.input / 32;
    let weight_groups = linear.input / 128;
    (0..linear.out)
        .into_par_iter()
        .map(|o| {
            let mut total = 0.0f32;
            for g in 0..groups {
                let start = o * linear.input / 8 + g * 4;
                let w = u32::from_le_bytes(linear.packed[start..start + 4].try_into().unwrap());
                let p = &x.planes[g * 4..g * 4 + 4];
                let matches = w ^ p[3];
                let count = (matches & p[0]).count_ones() as i32
                    + 2 * (matches & p[1]).count_ones() as i32
                    + 4 * (matches & p[2]).count_ones() as i32;
                let wg = o * weight_groups + g / 4;
                if x.a4_kernel == Kernel::SignmagExact {
                    let dot = count - x.a4_offsets[g];
                    total += (linear.scales[wg] * dot as f32
                        + linear.biases[wg] * x.sums[g] as f32)
                        * x.scales[g];
                } else {
                    let dot = 2 * count - x.a4_offsets[g];
                    total = (dot as f32 * x.scales[g]).mul_add(linear.scales[wg] * 0.5, total);
                }
            }
            total
        })
        .collect()
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn mv_tiled_neon(linear: &Linear, x: &PackedActivation) -> Vec<f32> {
    let mut output = vec![0.0f32; linear.out];
    let groups = linear.input / 128;
    output
        .par_chunks_mut(4)
        .with_min_len(16)
        .enumerate()
        .for_each(|(tile, out)| {
            let o = tile * 4;
            unsafe {
                if out.len() == 4 {
                    crate::bitplane_neon::dot_signmag_exact4_g32(
                        linear.packed.as_ptr().add(o * linear.input / 8),
                        linear.scales.as_ptr().add(o * groups),
                        linear.biases.as_ptr().add(o * groups),
                        x.planes.as_ptr(),
                        x.scales.as_ptr(),
                        x.sums.as_ptr(),
                        x.a4_offsets.as_ptr(),
                        groups,
                        out.as_mut_ptr(),
                    );
                } else {
                    for (j, y) in out.iter_mut().enumerate() {
                        *y = crate::bitplane_neon::dot_signmag_exact_g32(
                            linear.packed.as_ptr().add((o + j) * linear.input / 8),
                            linear.scales.as_ptr().add((o + j) * groups),
                            linear.biases.as_ptr().add((o + j) * groups),
                            x.planes.as_ptr(),
                            x.scales.as_ptr(),
                            x.sums.as_ptr(),
                            x.a4_offsets.as_ptr(),
                            groups,
                        );
                    }
                }
            }
        });
    output
}

/// Runs the production packers and row kernels in the same release executable.
pub(crate) fn microbench() -> anyhow::Result<()> {
    use std::{hint::black_box, time::Instant};
    let mut rows = Vec::new();
    for input in [2048usize, 6144] {
        let values: Vec<f32> = (0..input)
            .map(|i| ((i * 7919 % 65521) as f32 - 32760.0) / 32760.0)
            .collect();
        let weights: Vec<u8> = (0..input / 8 * 4).map(|i| (i * 79 + 37) as u8).collect();
        let scales = vec![0.03125f32; input / 128 * 4];
        let biases = vec![-0.015625f32; input / 128 * 4];
        let kernels = [
            (Kernel::Reference, 1),
            (Kernel::SignmagExact, 1),
            (Kernel::SignmagFast, 1),
            (Kernel::SignmagExact, 4),
        ];
        let make = |kernel| {
            if kernel == Kernel::Reference {
                PackedActivation::quantize(black_box(&values), 32, 4, crate::A2Scheme::Escape)
            } else {
                quantize(black_box(&values), kernel)
            }
        };
        for round in 0..7 {
            for j in 0..kernels.len() {
                let (kernel, tile) = kernels[(j + round) % kernels.len()];
                black_box(make(kernel));
                let iterations = 2000;
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(make(kernel));
                }
                let packing_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
                let packed = make(kernel);
                #[cfg(target_arch = "aarch64")]
                let row_kernel_ns = {
                    let iterations = 20000;
                    let start = Instant::now();
                    for _ in 0..iterations {
                        let p = black_box(&packed);
                        let value = unsafe {
                            if tile == 4 {
                                let mut out = [0.0f32; 4];
                                crate::bitplane_neon::dot_signmag_exact4_g32(
                                    weights.as_ptr(),
                                    scales.as_ptr(),
                                    biases.as_ptr(),
                                    p.planes.as_ptr(),
                                    p.scales.as_ptr(),
                                    p.sums.as_ptr(),
                                    p.a4_offsets.as_ptr(),
                                    input / 128,
                                    out.as_mut_ptr(),
                                );
                                black_box(out)[0]
                            } else {
                                match kernel {
                                    Kernel::Reference => crate::bitplane_neon::dot_g32(
                                        weights.as_ptr(),
                                        scales.as_ptr(),
                                        biases.as_ptr(),
                                        p.planes.as_ptr(),
                                        p.scales.as_ptr(),
                                        p.sums.as_ptr(),
                                        input / 128,
                                    ),
                                    Kernel::SignmagExact => {
                                        crate::bitplane_neon::dot_signmag_exact_g32(
                                            weights.as_ptr(),
                                            scales.as_ptr(),
                                            biases.as_ptr(),
                                            p.planes.as_ptr(),
                                            p.scales.as_ptr(),
                                            p.sums.as_ptr(),
                                            p.a4_offsets.as_ptr(),
                                            input / 128,
                                        )
                                    }
                                    Kernel::SignmagFast => {
                                        crate::bitplane_neon::dot_signmag_fast_g32(
                                            weights.as_ptr(),
                                            scales.as_ptr(),
                                            p.planes.as_ptr(),
                                            p.scales.as_ptr(),
                                            p.a4_offsets.as_ptr(),
                                            input / 128,
                                        )
                                    }
                                }
                            }
                        };
                        black_box(value);
                    }
                    start.elapsed().as_nanos() as f64 / iterations as f64 / tile as f64
                };
                #[cfg(not(target_arch = "aarch64"))]
                let row_kernel_ns = serde_json::Value::Null;
                rows.push(serde_json::json!({"input":input, "kernel":if tile==4 { "signmag-exact-tile4" } else {kernel.name()},
                    "output_rows":tile, "row_time_normalized_per_output":true, "round":round+1, "packing_ns":packing_ns, "row_kernel_ns":row_kernel_ns}));
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "kind":"production packing and row kernel microbenchmark; includes packing allocations; hot cache",
            "packing_iterations":2000, "row_iterations":20000, "trials":rows
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{A2Scheme, PackedActivation};

    fn next(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 32) as u32
    }

    fn decode(p: &PackedActivation, i: usize) -> i32 {
        let planes = &p.planes[(i / 32) * 4..];
        let bit = i % 32;
        let low = ((planes[0] >> bit) & 1) as i32
            + 2 * ((planes[1] >> bit) & 1) as i32
            + 4 * ((planes[2] >> bit) & 1) as i32;
        let sign = ((planes[3] >> bit) & 1) as i32;
        if p.a4_kernel == Kernel::Reference {
            low - 8 * sign
        } else {
            low * (1 - 2 * sign)
        }
    }

    #[test]
    fn all_codes_and_weight_signs() {
        for q in -7i32..=7 {
            for w in 0..=1 {
                let m = q.unsigned_abs();
                let matches = w ^ u32::from(q < 0);
                let count = matches * m;
                assert_eq!(2 * count as i32 - m as i32, (2 * w as i32 - 1) * q);
                assert_eq!(count as i32 - (-q).max(0), w as i32 * q);
            }
        }
    }

    #[test]
    #[should_panic(expected = "signmag-fast requires centered W1 weights")]
    fn fast_rejects_non_centered_weights() {
        let linear = Linear {
            out: 1,
            input: 128,
            packed: vec![0; 16],
            scales: vec![1.0],
            biases: vec![0.25],
            binary_centered: false,
        };
        linear.mv_quantized(&quantize(&[0.0; 128], Kernel::SignmagFast));
    }

    #[test]
    fn packing_preserves_reference_quantized_values() {
        let mut state = 20261010;
        for case in 0..40 {
            let values: Vec<f32> = (0..6144)
                .map(|i| match case {
                    0 => 0.0,
                    1 => -0.0,
                    2 => (i as i32 % 15 - 7) as f32,
                    3 => {
                        if i % 2 == 0 {
                            -1.0e30
                        } else {
                            1.0e30
                        }
                    }
                    _ => {
                        next(&mut state) as i32 as f32 / i32::MAX as f32
                            * if i % 17 == 0 { 128.0 } else { 1.0 }
                    }
                })
                .collect();
            let reference = PackedActivation::quantize(&values, 32, 4, A2Scheme::Escape);
            for kernel in [Kernel::SignmagExact, Kernel::SignmagFast] {
                let packed = quantize(&values, kernel);
                assert_eq!(reference.sums, packed.sums);
                assert_eq!(reference.scales, packed.scales);
                for i in 0..values.len() {
                    assert_eq!(decode(&reference, i), decode(&packed, i));
                }
                for g in 0..values.len() / 32 {
                    let expected = (g * 32..g * 32 + 32)
                        .map(|i| {
                            let q = decode(&packed, i);
                            if kernel == Kernel::SignmagExact {
                                (-q).max(0)
                            } else {
                                q.abs()
                            }
                        })
                        .sum::<i32>();
                    assert_eq!(packed.a4_offsets[g], expected);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_exact_and_fast_match_dense_random_weights() {
        let mut state = 81723;
        let mut max_normalized_error = 0.0f32;
        let mut compared = 0;
        // Includes odd numbers of 128-weight blocks to exercise the fast tail.
        for input in [128, 384, 2048, 6144] {
            for case in 0..24 {
                let values: Vec<f32> = (0..input)
                    .map(|i| match case {
                        0 => 0.0,
                        1 => (i as i32 % 15 - 7) as f32,
                        2 => {
                            if i % 32 == 0 {
                                1000.0
                            } else {
                                -0.01
                            }
                        }
                        _ => next(&mut state) as i32 as f32 / i32::MAX as f32,
                    })
                    .collect();
                let packed: Vec<u8> = (0..input / 8 * 17)
                    .map(|_| next(&mut state) as u8)
                    .collect();
                let scales: Vec<f32> = (0..input / 128 * 17)
                    .map(|_| (next(&mut state) % 4096 + 1) as f32 / 65536.0)
                    .collect();
                let biases = scales.iter().map(|s| -s * 0.5).collect();
                let linear = Linear {
                    out: 17,
                    input,
                    packed,
                    scales,
                    biases,
                    binary_centered: true,
                };
                let reference = PackedActivation::quantize(&values, 32, 4, A2Scheme::Escape);
                let exact = quantize(&values, Kernel::SignmagExact);
                let fast = quantize(&values, Kernel::SignmagFast);
                let a = linear.mv_quantized(&reference);
                let b = linear.mv_quantized(&exact);
                let mut tiled = quantize(&values, Kernel::SignmagExact);
                tiled.a4_row_tile = 4;
                let tiled_output = linear.mv_quantized(&tiled);
                assert_eq!(
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    tiled_output.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                let scalar = mv_scalar(&linear, &exact);
                let c = linear.mv_quantized(&fast);
                for (((a, b), s), c) in a.iter().zip(&b).zip(&scalar).zip(&c) {
                    assert_eq!(a.to_bits(), b.to_bits());
                    assert_eq!(a.to_bits(), s.to_bits());
                    let error = (a - c).abs() / (1.0 + a.abs());
                    max_normalized_error = max_normalized_error.max(error);
                    assert!(error <= 2.0e-4, "reference={a}, fast={c}, error={error}");
                    compared += 1;
                }
            }
        }
        println!(
            "A4_LINEAR_CHECK outputs={compared} max_normalized_error={max_normalized_error:e}"
        );
    }
}
