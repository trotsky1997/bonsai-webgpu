use super::*;

#[test]
fn parallel_escape_packing_matches_reference() {
    for threads in [1, 4, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for group_size in [32, 64, 128] {
                for seed in 0..12u64 {
                    let mut state = seed + 1;
                    let values: Vec<f32> = (0..4096)
                        .map(|i| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            let value = (state >> 32) as i32 as f32 / i32::MAX as f32;
                            if i < 128 || state & 7 == 0 {
                                0.0
                            } else if i % 32 == 0 {
                                value * 64.0
                            } else {
                                value
                            }
                        })
                        .collect();
                    let actual = PackedActivation::quantize_escape(&values, group_size);
                    let words = group_size / 32;
                    for (g, chunk) in values.chunks_exact(group_size).enumerate() {
                        let (alpha, beta) = crate::escape_fit::tests::reference_scales(chunk);
                        let mut planes = vec![0u32; words * 2];
                        let (mut sum, mut count) = (0i32, 0i32);
                        for (i, &value) in chunk.iter().enumerate() {
                            let mut q = if alpha == 0.0 {
                                0i8
                            } else {
                                (value / alpha).round().clamp(-1.0, 1.0) as i8
                            };
                            if (value - beta).abs() < (value - alpha * q as f32).abs() {
                                q = -2;
                                count += 1;
                            }
                            sum += q as i32;
                            let code = (q as u8) & 3;
                            for bit in 0..2 {
                                planes[bit * words + i / 32] |=
                                    (((code >> bit) & 1) as u32) << (i % 32);
                            }
                        }
                        assert_eq!(actual.scales[g].to_bits(), alpha.to_bits());
                        assert_eq!(
                            actual.corrections[g].to_bits(),
                            (beta + 2.0 * alpha).to_bits()
                        );
                        assert_eq!(actual.sums[g], sum);
                        assert_eq!(actual.escape_counts[g], count);
                        assert_eq!(&actual.planes[g * words * 2..(g + 1) * words * 2], planes);
                    }
                }
            }
        });
    }
}
