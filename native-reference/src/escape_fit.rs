//! Least-squares {0,-alpha,+alpha,beta} fit with the original candidate order.
//! Keep FP64 sums in the same order, including ties. Reciprocal products only
//! filter comparisons; close scores use the original divisions.

const RECIP: [f64; 129] = {
    let mut values = [0.0; 129];
    let mut i = 1;
    while i < values.len() {
        values[i] = 1.0 / i as f64;
        i += 1;
    }
    values
};

#[derive(Clone, Copy, Default)]
struct Core {
    sum: f64,
    square: f64,
    count: usize,
    estimate: f64,
}

// All nonzero scores are positive and normal for finite FP32 inputs. This guard
// exceeds the rounding error of a reciprocal product, division and positive
// sum. Therefore well-separated estimates have the same ordering as division.
#[inline(always)]
fn separated(a: f64, b: f64) -> bool {
    (a - b).abs() > a.max(b) * (16.0 * f64::EPSILON)
}

impl Core {
    #[inline(always)]
    fn exact_score(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.square / self.count as f64
        }
    }

    #[inline(always)]
    fn consider(&mut self, sum: f64, count: usize) {
        let square = sum * sum;
        let estimate = square * RECIP[count];
        let better = if separated(estimate, self.estimate) {
            estimate > self.estimate
        } else {
            square / count as f64 > self.exact_score()
        };
        if better {
            *self = Self {
                sum,
                square,
                count,
                estimate,
            };
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Fit {
    core: Core,
    escape_sum: f64,
    escape_count: usize,
    negative: bool,
    estimate: f64,
}

impl Fit {
    #[inline(always)]
    fn exact_score(self) -> f64 {
        self.core.exact_score()
            + if self.escape_count == 0 {
                0.0
            } else {
                self.escape_sum * self.escape_sum / self.escape_count as f64
            }
    }

    #[inline(always)]
    fn consider(&mut self, core: Core, escape_sum: f64, escape_count: usize, negative: bool) {
        let estimate = core.estimate + (escape_sum * escape_sum) * RECIP[escape_count];
        let candidate = Self {
            core,
            escape_sum,
            escape_count,
            negative,
            estimate,
        };
        let better = if separated(estimate, self.estimate) {
            estimate > self.estimate
        } else {
            candidate.exact_score() > self.exact_score()
        };
        if better {
            *self = candidate;
        }
    }
}

pub fn escape_scales(values: &[f32]) -> (f32, f32) {
    assert!(!values.is_empty() && values.len() <= 128);
    let mut sorted = [(0.0f64, false); 128];
    for (dst, &x) in sorted.iter_mut().zip(values) {
        *dst = (x.abs() as f64, x < 0.0);
    }
    let sorted = &mut sorted[..values.len()];
    sorted.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let mut core = Core::default();
    let mut sum = 0.0;
    for (i, &(m, _)) in sorted.iter().enumerate() {
        sum += m;
        core.consider(sum, i + 1);
    }
    let mut best = Fit {
        core,
        estimate: core.estimate,
        ..Fit::default()
    };

    for negative in [false, true] {
        let (mut escape_sum, mut escape_count) = (0.0, 0usize);
        let (mut opposite_sum, mut opposite_count) = (0.0, 0usize);
        let mut opposite_core = Core::default();
        for (i, &(m, sign)) in sorted.iter().enumerate() {
            if sign != negative || m == 0.0 {
                opposite_sum += m;
                opposite_count += 1;
                opposite_core.consider(opposite_sum, opposite_count);
                continue;
            }
            escape_sum += m;
            escape_count += 1;
            // The first k same-sign entries have all been removed. Before
            // their last position only the opposite-sign prefix survives;
            // after it every element remains. Both sums keep baseline order.
            let mut candidate_core = opposite_core;
            let (mut sum, mut count) = (opposite_sum, opposite_count);
            for &(remaining, _) in &sorted[i + 1..] {
                sum += remaining;
                count += 1;
                candidate_core.consider(sum, count);
            }
            best.consider(candidate_core, escape_sum, escape_count, negative);
        }
    }
    let alpha = if best.core.count == 0 {
        0.0
    } else {
        best.core.sum / best.core.count as f64
    };
    let beta = if best.escape_count == 0 {
        0.0
    } else {
        best.escape_sum / best.escape_count as f64 * if best.negative { -1.0 } else { 1.0 }
    };
    (alpha as f32, beta as f32)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn reference_scales(values: &[f32]) -> (f32, f32) {
        let mut sorted = [(0.0f64, false); 128];
        for (dst, &x) in sorted.iter_mut().zip(values) {
            *dst = (x.abs() as f64, x < 0.0);
        }
        let sorted = &mut sorted[..values.len()];
        sorted.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let (mut best_score, mut alpha, mut beta) = (0.0f64, 0.0, 0.0);
        for negative in [false, true] {
            let count = sorted
                .iter()
                .filter(|x| x.1 == negative && x.0 > 0.0)
                .count();
            let mut escape_sum = 0.0;
            for k in 0..=count {
                let (mut skipped, mut sum, mut remaining) = (0, 0.0, 0usize);
                let (mut core_score, mut core_alpha) = (0.0, 0.0);
                for &(m, sign) in sorted.iter() {
                    if sign == negative && skipped < k {
                        skipped += 1;
                        continue;
                    }
                    sum += m;
                    remaining += 1;
                    let score = sum * sum / remaining as f64;
                    if score > core_score {
                        core_score = score;
                        core_alpha = sum / remaining as f64;
                    }
                }
                let score = core_score
                    + if k > 0 {
                        escape_sum * escape_sum / k as f64
                    } else {
                        0.0
                    };
                if score > best_score {
                    best_score = score;
                    alpha = core_alpha;
                    beta = if k > 0 {
                        escape_sum / k as f64 * if negative { -1.0 } else { 1.0 }
                    } else {
                        0.0
                    };
                }
                if k < count {
                    escape_sum += sorted
                        .iter()
                        .filter(|x| x.1 == negative && x.0 > 0.0)
                        .nth(k)
                        .unwrap()
                        .0;
                }
            }
        }
        (alpha as f32, beta as f32)
    }

    fn check(values: &[f32]) {
        let expected = reference_scales(values);
        let actual = escape_scales(values);
        assert_eq!(
            (actual.0.to_bits(), actual.1.to_bits()),
            (expected.0.to_bits(), expected.1.to_bits()),
            "input={values:?}"
        );
    }

    #[test]
    fn boundary_and_tie_parity() {
        for n in [32, 64, 128] {
            for c in [0.0, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::MAX] {
                check(&vec![c; n]);
            }
            for pattern in [
                vec![1.0, -1.0],
                vec![0.0, 1.0, -1.0],
                vec![10.0, 4.0, 4.0, 4.0],
                vec![1.0, -1.0, 8.0, -8.0, 0.0],
            ] {
                check(
                    &(0..n)
                        .map(|i| pattern[i % pattern.len()])
                        .collect::<Vec<_>>(),
                );
            }
        }
    }

    #[test]
    fn randomized_bitwise_scale_parity() {
        let mut state = 0xd061_135a_a901_b73du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for n in [32, 64, 128] {
            for case in 0..12000 {
                let values = (0..n)
                    .map(|_| {
                        let r = next();
                        match case % 6 {
                            0 => ((r >> 32) as i32 as f32) / (i32::MAX as f32),
                            1 => {
                                if r & 7 == 0 {
                                    ((r >> 32) as i32 as f32) * 1e-6
                                } else {
                                    ((r >> 32) as i32 as f32) * 1e-9
                                }
                            }
                            2 => (r % 9) as f32 - 4.0,
                            3 => f32::from_bits(
                                (r as u32 & 0x807fffff) | (((r >> 32) % 254 + 1) as u32) << 23,
                            ),
                            4 => {
                                if r & 3 == 0 {
                                    0.0
                                } else {
                                    (r % 31) as f32 / 7.0
                                }
                            }
                            _ => -((r % 31) as f32 / 7.0),
                        }
                    })
                    .collect::<Vec<_>>();
                check(&values);
            }
        }
    }
}
