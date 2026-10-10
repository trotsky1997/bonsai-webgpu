//! The fourth A2 code is a marker. Read its value from a compact signed INT4
//! stream in increasing position order. Fixed escape count eliminates offsets.
use super::*;

pub(crate) struct Sidecar {
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
    pub nibbles: Vec<u8>,
    pub bytes_per_group: usize,
    pub per32: usize,
    pub cache: Vec<RowCache>,
    pub cache_mode: u8,
}

pub(crate) fn quantize(values: &[f32], group_size: usize, per32: usize) -> PackedActivation {
    let groups = values.len() / group_size;
    let words = group_size / 32;
    let k = per32 * words;
    let bytes = k.div_ceil(2);
    let mut result = PackedActivation {
        planes: vec![0; groups * words * 2],
        scales: vec![0.0; groups],
        sums: vec![0; groups],
        corrections: Vec::new(),
        escape_counts: Vec::new(),
        group_size,
        bits: 2,
        fast_reduction: false,
        a4_kernel: crate::a4::Kernel::Reference,
        a4_offsets: Vec::new(),
        a4_row_tile: 1,
        balanced: false,
        sidecar: Some(Sidecar {
            scales: vec![0.0; groups],
            sums: vec![0; groups],
            nibbles: vec![0; groups * bytes],
            bytes_per_group: bytes,
            per32,
            cache: Vec::new(),
            cache_mode: 0,
        }),
    };
    let side = result.sidecar.as_mut().unwrap();
    result
        .planes
        .par_chunks_mut(2 * words)
        .zip(result.scales.par_iter_mut())
        .zip(result.sums.par_iter_mut())
        .zip(side.scales.par_iter_mut())
        .zip(side.sums.par_iter_mut())
        .zip(side.nibbles.par_chunks_mut(bytes))
        .zip(values.par_chunks_exact(group_size))
        .with_min_len(4)
        .for_each(|((((((planes, scale), sum), s4), sum4), nibbles), chunk)| {
            assert!(chunk.iter().all(|x| x.is_finite()), "non-finite activation");
            let mut sorted = [(0.0f32, 0usize); 128];
            for (i, &x) in chunk.iter().enumerate() {
                sorted[i] = (x.abs(), i);
            }
            let sorted = &mut sorted[..group_size];
            sorted.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut mask = 0u128;
            for &(_, i) in &sorted[..k] {
                mask |= 1u128 << i;
            }
            // Least-squares ternary fit of the remaining values: one prefix scan.
            let (mut total, mut score, mut alpha) = (0.0f64, 0.0f64, 0.0f64);
            for (i, &(m, _)) in sorted[k..].iter().enumerate() {
                total += m as f64;
                let candidate = total * total / (i + 1) as f64;
                if candidate > score {
                    score = candidate;
                    alpha = total / (i + 1) as f64;
                }
            }
            *scale = alpha as f32;
            let max_abs = sorted[0].0;
            *s4 = max_abs / 7.0;
            let mut cursor = 0;
            for (i, &x) in chunk.iter().enumerate() {
                let escaped = mask & (1u128 << i) != 0;
                let q = if escaped {
                    let q4 = if max_abs == 0.0 {
                        0
                    } else {
                        ((x / max_abs) * 7.0).round().clamp(-7.0, 7.0) as i8
                    };
                    nibbles[cursor / 2] |= ((q4 as u8) & 15) << ((cursor % 2) * 4);
                    *sum4 += q4 as i32;
                    cursor += 1;
                    -2i8
                } else if *scale == 0.0 {
                    0
                } else {
                    (x / *scale).round().clamp(-1.0, 1.0) as i8
                };
                if !escaped {
                    *sum += q as i32;
                }
                let word = i / 32;
                planes[word] |= u32::from(q == -1 || q == 1) << (i % 32);
                planes[words + word] |= u32::from(q < 0) << (i % 32);
            }
            debug_assert_eq!(cursor, k);
        });
    result
}

#[inline(always)]
pub(crate) fn nibble(byte: u8, index: usize) -> i32 {
    let q = (byte >> ((index & 1) * 4)) & 15;
    ((q as i8) << 4 >> 4) as i32
}

pub(crate) fn mv_scalar(linear: &Linear, x: &PackedActivation) -> Vec<f32> {
    let side = x.sidecar.as_ref().unwrap();
    let groups = linear.input / x.group_size;
    let weight_groups = linear.input / 128;
    let words = x.group_size / 32;
    (0..linear.out)
        .into_par_iter()
        .with_min_len(64)
        .map(|o| {
            let mut total = 0.0f32;
            for g in 0..groups {
                let planes = &x.planes[g * 2 * words..(g + 1) * 2 * words];
                let mut dot = 0;
                let mut dot4 = 0;
                let mut cursor = 0;
                for word in 0..words {
                    let start = o * linear.input / 8 + g * x.group_size / 8 + word * 4;
                    let w = u32::from_le_bytes(linear.packed[start..start + 4].try_into().unwrap());
                    let low = planes[word];
                    let high = planes[words + word];
                    dot += (w & low).count_ones() as i32 - 2 * (w & low & high).count_ones() as i32;
                    let mut mask = high & !low;
                    while mask != 0 {
                        let i = mask.trailing_zeros();
                        let q4 =
                            nibble(side.nibbles[g * side.bytes_per_group + cursor / 2], cursor);
                        dot4 += q4 * ((w >> i) & 1) as i32;
                        cursor += 1;
                        mask &= mask - 1;
                    }
                }
                let wg = o * weight_groups + g * x.group_size / 128;
                total += x.scales[g]
                    * (linear.scales[wg] * dot as f32 + linear.biases[wg] * x.sums[g] as f32);
                total += side.scales[g]
                    * (linear.scales[wg] * dot4 as f32 + linear.biases[wg] * side.sums[g] as f32);
            }
            total
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_stream_and_dot_match_dense_reconstruction() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let mut state = 0xa37c_93d5_821d_116bu64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            for group_size in [32, 64, 128] {
                for k in [1, 2, 4, 8] {
                    for case in 0..8 {
                        let values: Vec<f32> = (0..512)
                            .map(|i| {
                                let value = (next() >> 32) as i32 as f32 / i32::MAX as f32;
                                match case {
                                    0 => 0.0,
                                    1 => 1.0,
                                    2 => -1.0,
                                    3 => (i % 5) as f32 - 2.0,
                                    _ if i % 32 < 2 => value * 30.0,
                                    _ => value,
                                }
                            })
                            .collect();
                        let packed = quantize(&values, group_size, k);
                        let side = packed.sidecar.as_ref().unwrap();
                        let words = group_size / 32;
                        let mut decoded = Vec::new();
                        for (g, chunk) in values.chunks_exact(group_size).enumerate() {
                            let (mut cursor, mut sum, mut sum4) = (0, 0, 0);
                            for (i, &value) in chunk.iter().enumerate() {
                                let base = g * words * 2 + i / 32;
                                let low = (packed.planes[base] >> (i % 32)) & 1;
                                let high = (packed.planes[base + words] >> (i % 32)) & 1;
                                let q = low as i32 - 2 * high as i32;
                                let x = if q == -2 {
                                    let q4 = nibble(
                                        side.nibbles[g * side.bytes_per_group + cursor / 2],
                                        cursor,
                                    );
                                    let max = chunk.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                                    let expected = if max == 0.0 {
                                        0
                                    } else {
                                        ((value / max) * 7.0).round() as i32
                                    };
                                    assert_eq!(q4, expected);
                                    sum4 += q4;
                                    cursor += 1;
                                    q4 as f64 * side.scales[g] as f64
                                } else {
                                    sum += q;
                                    q as f64 * packed.scales[g] as f64
                                };
                                decoded.push(x);
                            }
                            assert_eq!(cursor, k * words);
                            assert_eq!(sum, packed.sums[g]);
                            assert_eq!(sum4, side.sums[g]);
                        }
                        let linear = Linear {
                            out: 5,
                            input: values.len(),
                            packed: (0..5 * values.len() / 8).map(|_| next() as u8).collect(),
                            scales: (0..20).map(|i| (i + 1) as f32 / 20.0).collect(),
                            biases: (0..20).map(|i| -(i + 2) as f32 / 30.0).collect(),
                            binary_centered: false,
                        };
                        let actual = mv_scalar(&linear, &packed);
                        for (o, &result) in actual.iter().enumerate() {
                            let mut reference = 0.0f64;
                            let mut magnitude = 0.0f64;
                            for (i, &x) in decoded.iter().enumerate() {
                                let bit =
                                    (linear.packed[o * values.len() / 8 + i / 8] >> (i % 8)) & 1;
                                let wg = o * 4 + i / 128;
                                let weight = linear.scales[wg] as f64 * bit as f64
                                    + linear.biases[wg] as f64;
                                reference += weight * x;
                                magnitude += (weight * x).abs();
                            }
                            assert!((result as f64 - reference).abs() <= 2e-6 * magnitude.max(1.0));
                        }
                        #[cfg(target_arch = "aarch64")]
                        if group_size == 32 {
                            let neon = linear.mv_quantized_neon(&packed);
                            for (a, b) in actual.iter().zip(neon) {
                                assert_eq!(a.to_bits(), b.to_bits());
                            }
                        }
                    }
                }
            }
        });
    }
}

#[derive(Clone, Default)]
pub(crate) struct RowCache {
    pub shifts: [[i32; 4]; 2],
    pub data: [[u32; 4]; 2],
    pub nonzero: [i32; 4],
}

pub(crate) fn prepare_cache(packed: &mut PackedActivation, mode: u8) {
    let side = packed.sidecar.as_mut().unwrap();
    assert_eq!(packed.group_size, 32);
    assert_eq!(side.per32, 2);
    side.cache = vec![RowCache::default(); packed.scales.len() / 4];
    side.cache_mode = mode;
    side.cache
        .par_iter_mut()
        .enumerate()
        .for_each(|(wg, cache)| {
            for lane in 0..4 {
                let g = wg * 4 + lane;
                cache.nonzero[lane] = packed.planes[g * 2].count_ones() as i32;
                let mut mask = packed.planes[g * 2 + 1] & !packed.planes[g * 2];
                for rank in 0..2 {
                    let position = mask.trailing_zeros();
                    mask &= mask - 1;
                    let q = nibble(side.nibbles[g], rank);
                    cache.shifts[rank][lane] = -(position as i32);
                    cache.data[rank][lane] = if mode == 2 {
                        (q as f32 * side.scales[g]).to_bits()
                    } else {
                        q as u32
                    };
                }
            }
        });
}

pub(crate) fn quantize_calibrated(
    values: &[f32],
    site: usize,
    cal: &crate::calibration::Calibration,
) -> PackedActivation {
    let groups = values.len() / 32;
    let mut packed = PackedActivation {
        planes: vec![0; groups * 2],
        scales: vec![0.0; groups],
        sums: vec![0; groups],
        corrections: Vec::new(),
        escape_counts: Vec::new(),
        group_size: 32,
        bits: 2,
        fast_reduction: false,
        a4_kernel: crate::a4::Kernel::Reference,
        a4_offsets: Vec::new(),
        a4_row_tile: 1,
        balanced: false,
        sidecar: Some(Sidecar {
            scales: vec![0.0; groups],
            sums: vec![0; groups],
            nibbles: vec![0; groups],
            bytes_per_group: 1,
            per32: 2,
            cache: Vec::new(),
            cache_mode: 0,
        }),
    };
    let side = packed.sidecar.as_mut().unwrap();
    packed
        .planes
        .par_chunks_mut(2)
        .zip(packed.scales.par_iter_mut())
        .zip(packed.sums.par_iter_mut())
        .zip(side.scales.par_iter_mut())
        .zip(side.sums.par_iter_mut())
        .zip(side.nibbles.par_iter_mut())
        .zip(values.par_chunks_exact(32))
        .enumerate()
        .with_min_len(4)
        .for_each(
            |(g, ((((((planes, scale), sum), s4), sum4), byte), chunk))| {
                assert!(chunk.iter().all(|x| x.is_finite()), "non-finite activation");
                let threshold = if cal.rule.starts_with("threshold") {
                    if cal.per_group {
                        cal.sites[site].thresholds[g]
                    } else {
                        cal.sites[site].shared_threshold
                    }
                } else {
                    0.6
                };
                let (indices, features, max) =
                    crate::calibration::features_with_threshold(chunk, threshold);
                let coeff = cal.coefficients(site, g);
                let mut alpha = if cal.rule.starts_with("threshold") {
                    features[3]
                } else {
                    features.iter().zip(coeff).map(|(x, c)| x * c).sum::<f32>()
                } * cal.alpha_factor;
                let refinements = match cal.rule.as_str() {
                    "threshold-refine1" => 1,
                    "threshold-refine2" => 2,
                    _ => 0,
                };
                for _ in 0..refinements {
                    let mut sum = 0.0f64;
                    let mut count = 0;
                    for (i, &x) in chunk.iter().enumerate() {
                        if i != indices[0] && i != indices[1] && x.abs() >= alpha * 0.5 {
                            sum += x.abs() as f64;
                            count += 1;
                        }
                    }
                    alpha = if count > 0 {
                        (sum / count as f64) as f32
                    } else {
                        0.0
                    };
                }
                *scale = alpha.clamp(features[0], features[2]);
                *s4 = max / 7.0;
                let mut cursor = 0;
                for (i, &x) in chunk.iter().enumerate() {
                    let escaped = i == indices[0] || i == indices[1];
                    let q = if escaped {
                        let q4 = if max == 0.0 {
                            0
                        } else {
                            ((x / max) * 7.0).round().clamp(-7.0, 7.0) as i8
                        };
                        *byte |= ((q4 as u8) & 15) << (cursor * 4);
                        *sum4 += q4 as i32;
                        cursor += 1;
                        -2i8
                    } else {
                        // Direct ternary thresholds: no division or rounding per value.
                        let threshold = *scale * 0.5;
                        if *scale == 0.0 {
                            0
                        } else if x >= threshold {
                            1
                        } else if x <= -threshold {
                            -1
                        } else {
                            0
                        }
                    };
                    if !escaped {
                        *sum += q as i32;
                    }
                    planes[0] |= u32::from(q == 1 || q == -1) << i;
                    planes[1] |= u32::from(q < 0) << i;
                }
            },
        );
    packed
}

#[cfg(test)]
mod calibration_tests {
    use super::*;
    #[test]
    fn calibrated_codes_and_both_cached_kernels() {
        let mut state = 96123457u64;
        for seed in 0..24 {
            let values: Vec<f32> = (0..512)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let x = (state >> 32) as i32 as f32 / i32::MAX as f32;
                    if seed == 0 {
                        0.0
                    } else if seed == 1 {
                        1.0
                    } else if i % 32 < 2 {
                        x * 32.0
                    } else {
                        x
                    }
                })
                .collect();
            let cal = crate::calibration::Calibration {
                schema: "bonsai-sidecar-calibration-v1".into(),
                model_fingerprint: String::new(),
                corpus_fingerprint: String::new(),
                activation_group_size: 32,
                escapes_per_group: 2,
                feature_names: Vec::new(),
                alpha_factor: 1.0,
                per_group: true,
                rule: "regression".into(),
                tokens: 100,
                sites: vec![crate::calibration::Site {
                    name: "test".into(),
                    shared: [0.0, 1.2, 0.0, 0.0],
                    shared_threshold: 0.6,
                    thresholds: vec![0.6; 16],
                    coefficients: vec![[0.0, 1.2, 0.0, 0.0]; 16],
                    observations_per_group: 100,
                }],
            };
            let mut packed = quantize_calibrated(&values, 0, &cal);
            let side = packed.sidecar.as_ref().unwrap();
            for (g, chunk) in values.chunks_exact(32).enumerate() {
                let (indices, _, _) = crate::calibration::features(chunk);
                let mask = (1u32 << indices[0]) | (1u32 << indices[1]);
                assert_eq!(packed.planes[g * 2 + 1] & !packed.planes[g * 2], mask);
                let (mut sum, mut sum4, mut cursor) = (0, 0, 0);
                for (i, &x) in chunk.iter().enumerate() {
                    let q = ((packed.planes[g * 2] >> i) & 1) as i32
                        - 2 * ((packed.planes[g * 2 + 1] >> i) & 1) as i32;
                    if mask & (1 << i) != 0 {
                        assert_eq!(q, -2);
                        sum4 += nibble(side.nibbles[g], cursor);
                        cursor += 1;
                    } else {
                        sum += q;
                        assert!(
                            (x - packed.scales[g] * q as f32).abs()
                                <= x.abs().min((x.abs() - packed.scales[g]).abs()) + 1e-5
                        );
                    }
                }
                assert_eq!(sum, packed.sums[g]);
                assert_eq!(sum4, side.sums[g]);
                assert_eq!(cursor, 2);
            }
            let linear = Linear {
                out: 7,
                input: 512,
                packed: (0..7 * 64)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state as u8
                    })
                    .collect(),
                scales: (0..28).map(|i| (i + 1) as f32 * 0.03).collect(),
                biases: (0..28).map(|i| -(i + 1) as f32 * 0.03 * 0.5).collect(),
                binary_centered: true,
            };
            let mut a4 = PackedActivation::quantize(&values, 32, 4, A2Scheme::Sidecar(2));
            let a4_expected = linear.mv_quantized(&a4);
            a4.fast_reduction = true;
            #[cfg(target_arch = "aarch64")]
            for (a, b) in a4_expected.iter().zip(linear.mv_quantized(&a4)) {
                assert!((a - b).abs() < 2e-4 * (1.0 + a.abs()));
            }
            let expected = mv_scalar(&linear, &packed);
            prepare_cache(&mut packed, 1);
            #[cfg(target_arch = "aarch64")]
            for (a, b) in expected.iter().zip(linear.mv_quantized_neon(&packed)) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            prepare_cache(&mut packed, 2);
            #[cfg(target_arch = "aarch64")]
            for (a, b) in expected.iter().zip(linear.mv_quantized_neon(&packed)) {
                assert!((a - b).abs() < 2e-4 * (1.0 + a.abs()), "{a} {b}");
            }
        }
    }
}
