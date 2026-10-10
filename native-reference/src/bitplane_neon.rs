//! AArch64 W1A2/W1A4 g32 dot products. Weights remain packed at 1 bit.
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn popcount32(x: uint32x4_t) -> int32x4_t {
    vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(x)))))
}

/// Each SIMD lane is one 32-activation group within a 128-weight group.
///
/// Safety: weights contains groups*16 bytes; scales/biases groups floats;
/// planes groups*16 u32; a_scales/a_sums groups*4 elements. NEON is required.
#[inline(never)]
pub unsafe fn dot_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    groups: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for g in 0..groups {
        // LD4 transposes four g32 activation groups into four bitplanes.
        let a = vld4q_u32(planes.add(g * 16));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let p0 = popcount32(vandq_u32(w, a.0));
        let p1 = popcount32(vandq_u32(w, a.1));
        let p2 = popcount32(vandq_u32(w, a.2));
        let p3 = popcount32(vandq_u32(w, a.3));
        let dot = vsubq_s32(
            vaddq_s32(vaddq_s32(p0, vshlq_n_s32::<1>(p1)), vshlq_n_s32::<2>(p2)),
            vshlq_n_s32::<3>(p3),
        );
        let weighted = vmulq_n_f32(vcvtq_f32_s32(dot), *scales.add(g));
        let bias = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4))), *biases.add(g));
        let terms = vmulq_f32(vaddq_f32(weighted, bias), vld1q_f32(a_scales.add(g * 4)));
        // Keep the original scalar FP32 addition order for exact parity.
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<3>(terms);
    }
    sum
}

/// Safety: as dot_g32, but planes contains groups*8 u32, interleaved
/// two's-complement INT2 planes for each 32-element activation group.
#[inline]
pub unsafe fn dot_int2_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    groups: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for g in 0..groups {
        let a = vld2q_u32(planes.add(g * 8));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let dot = vsubq_s32(
            popcount32(vandq_u32(w, a.0)),
            vshlq_n_s32::<1>(popcount32(vandq_u32(w, a.1))),
        );
        let weighted = vmulq_n_f32(vcvtq_f32_s32(dot), *scales.add(g));
        let bias = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4))), *biases.add(g));
        let terms = vmulq_f32(vaddq_f32(weighted, bias), vld1q_f32(a_scales.add(g * 4)));
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<3>(terms);
    }
    sum
}

/// Safety: as dot_int2_g32. Corrections and escape_counts each contain
/// groups*4 elements. The fourth code (10) decodes to an independent beta.
#[inline]
pub unsafe fn dot_escape_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    corrections: *const f32,
    escape_counts: *const i32,
    groups: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for g in 0..groups {
        let a = vld2q_u32(planes.add(g * 8));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let dot = vsubq_s32(
            popcount32(vandq_u32(w, a.0)),
            vshlq_n_s32::<1>(popcount32(vandq_u32(w, a.1))),
        );
        let escape = popcount32(vandq_u32(w, vbicq_u32(a.1, a.0)));
        let weighted = vmulq_n_f32(vcvtq_f32_s32(dot), *scales.add(g));
        let bias = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4))), *biases.add(g));
        let terms = vmulq_f32(vaddq_f32(weighted, bias), vld1q_f32(a_scales.add(g * 4)));
        let escape_weighted = vmulq_n_f32(vcvtq_f32_s32(escape), *scales.add(g));
        let escape_bias = vmulq_n_f32(
            vcvtq_f32_s32(vld1q_s32(escape_counts.add(g * 4))),
            *biases.add(g),
        );
        let extra = vmulq_f32(
            vaddq_f32(escape_weighted, escape_bias),
            vld1q_f32(corrections.add(g * 4)),
        );
        // Match the generic kernel's alternating base/correction additions.
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<0>(extra);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<1>(extra);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<2>(extra);
        sum += vgetq_lane_f32::<3>(terms);
        sum += vgetq_lane_f32::<3>(extra);
    }
    sum
}

/// Safety: weights has groups*16 bytes; scales has groups elements;
/// planes groups*8 u32; c0/c1 each groups*4 floats. Centered binary weights.
#[inline]
pub unsafe fn dot_balanced_g32(
    weights: *const u8,
    scales: *const f32,
    planes: *const u32,
    c0: *const f32,
    c1: *const f32,
    groups: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let n = vdupq_n_s32(32);
    for g in 0..groups {
        let a = vld2q_u32(planes.add(g * 8));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let d0 = vsubq_s32(
            vshlq_n_s32::<1>(popcount32(vmvnq_u32(veorq_u32(w, a.0)))),
            n,
        );
        let d1 = vsubq_s32(
            vshlq_n_s32::<1>(popcount32(vmvnq_u32(veorq_u32(w, a.1)))),
            n,
        );
        let s = *scales.add(g) * 0.5;
        let t0 = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(d0), s), vld1q_f32(c0.add(g * 4)));
        let t1 = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(d1), s), vld1q_f32(c1.add(g * 4)));
        let terms = vaddq_f32(t0, t1);
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<3>(terms);
    }
    sum
}

/// Safety: base buffers as dot_int2_g32; side scale/sum groups*4 entries;
/// side nibbles groups*4*bytes_per_group bytes. Marker count fits its stream.
#[inline]
pub unsafe fn dot_sidecar_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    side_scales: *const f32,
    side_sums: *const i32,
    nibbles: *const u8,
    bytes_per_group: usize,
    per32: usize,
    groups: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for g in 0..groups {
        let a = vld2q_u32(planes.add(g * 8));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        // A marker contributes zero to the ternary base.
        let dot = vsubq_s32(
            popcount32(vandq_u32(w, a.0)),
            vshlq_n_s32::<1>(popcount32(vandq_u32(w, vandq_u32(a.0, a.1)))),
        );
        let terms = vmulq_f32(
            vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(dot), *scales.add(g)),
                vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4))), *biases.add(g)),
            ),
            vld1q_f32(a_scales.add(g * 4)),
        );
        // Load the compact streams for four groups, one stream per lane.
        let stream = nibbles.add(g * 4 * bytes_per_group);
        let packed = match bytes_per_group {
            1 => {
                let four = (stream as *const u32).read_unaligned();
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(four as u64)))))
            }
            2 => vreinterpretq_s32_u32(vmovl_u16(vld1_u16(stream as *const u16))),
            4 => vld1q_s32(stream as *const i32),
            _ => unreachable!(),
        };
        let mut markers = vbicq_u32(a.1, a.0);
        let mut dots = vdupq_n_s32(0);
        for rank in (0..per32).rev() {
            // Visit the highest marked position first in all four lanes.
            let right = vsubq_s32(vreinterpretq_s32_u32(vclzq_u32(markers)), vdupq_n_s32(31));
            let bit = vshlq_u32(vdupq_n_u32(1), vnegq_s32(right));
            let selected = vcgtq_u32(vandq_u32(w, bit), vdupq_n_u32(0));
            let q4 = vshrq_n_s32::<28>(vshlq_s32(packed, vdupq_n_s32(28 - 4 * rank as i32)));
            dots = vaddq_s32(dots, vandq_s32(q4, vreinterpretq_s32_u32(selected)));
            markers = veorq_u32(markers, bit);
        }
        let extra = vmulq_f32(
            vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(dots), *scales.add(g)),
                vmulq_n_f32(
                    vcvtq_f32_s32(vld1q_s32(side_sums.add(g * 4))),
                    *biases.add(g),
                ),
            ),
            vld1q_f32(side_scales.add(g * 4)),
        );
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<0>(extra);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<1>(extra);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<2>(extra);
        sum += vgetq_lane_f32::<3>(terms);
        sum += vgetq_lane_f32::<3>(extra);
    }
    sum
}

/// As dot_sidecar_g32 with predecoded positions/INT4 integers. Same FP32 order.
#[inline]
pub unsafe fn dot_cached_exact_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    side_scales: *const f32,
    side_sums: *const i32,
    cache: *const crate::sidecar::RowCache,
    groups: usize,
) -> f32 {
    let mut sum = 0.0;
    for g in 0..groups {
        let a = vld2q_u32(planes.add(g * 8));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let dot = vsubq_s32(
            popcount32(vandq_u32(w, a.0)),
            vshlq_n_s32::<1>(popcount32(vandq_u32(w, vandq_u32(a.0, a.1)))),
        );
        let terms = vmulq_f32(
            vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(dot), *scales.add(g)),
                vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4))), *biases.add(g)),
            ),
            vld1q_f32(a_scales.add(g * 4)),
        );
        let item = &*cache.add(g);
        let mut dots = vdupq_n_s32(0);
        for rank in 0..2 {
            let bits = vandq_u32(
                vshlq_u32(w, vld1q_s32(item.shifts[rank].as_ptr())),
                vdupq_n_u32(1),
            );
            let select = vnegq_s32(vreinterpretq_s32_u32(bits));
            dots = vaddq_s32(
                dots,
                vandq_s32(select, vld1q_s32(item.data[rank].as_ptr() as *const i32)),
            );
        }
        let extra = vmulq_f32(
            vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(dots), *scales.add(g)),
                vmulq_n_f32(
                    vcvtq_f32_s32(vld1q_s32(side_sums.add(g * 4))),
                    *biases.add(g),
                ),
            ),
            vld1q_f32(side_scales.add(g * 4)),
        );
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<0>(extra);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<1>(extra);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<2>(extra);
        sum += vgetq_lane_f32::<3>(terms);
        sum += vgetq_lane_f32::<3>(extra);
    }
    sum
}

/// Centered W1 affine weights; cached FP32 side values and vector reduction.
#[inline]
pub unsafe fn dot_cached_centered_g32(
    weights: *const u8,
    scales: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    _a_sums: *const i32,
    cache: *const crate::sidecar::RowCache,
    groups: usize,
) -> f32 {
    let mut total = vdupq_n_f32(0.0);
    for g in 0..groups {
        let a = vld2q_u32(planes.add(g * 8));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let item = &*cache.add(g);
        let matches = popcount32(vandq_u32(veorq_u32(w, a.1), a.0));
        let dot = vsubq_s32(vshlq_n_s32::<1>(matches), vld1q_s32(item.nonzero.as_ptr()));
        let mut terms = vmulq_f32(vcvtq_f32_s32(dot), vld1q_f32(a_scales.add(g * 4)));
        for rank in 0..2 {
            let sign = vshlq_n_u32::<31>(veorq_u32(
                vshlq_u32(w, vld1q_s32(item.shifts[rank].as_ptr())),
                vdupq_n_u32(1),
            ));
            let value = vreinterpretq_f32_u32(veorq_u32(vld1q_u32(item.data[rank].as_ptr()), sign));
            terms = vaddq_f32(terms, value);
        }
        total = vaddq_f32(total, vmulq_n_f32(terms, *scales.add(g) * 0.5));
    }
    vaddvq_f32(total)
}

/// Same centered-weight/vector-reduction optimization as the A2 cached path.
#[inline]
pub unsafe fn dot_a4_centered_g32(
    weights: *const u8,
    scales: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    groups: usize,
) -> f32 {
    let mut total = vdupq_n_f32(0.0);
    for g in 0..groups {
        let a = vld4q_u32(planes.add(g * 16));
        let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
        let dot = vsubq_s32(
            vaddq_s32(
                vaddq_s32(
                    popcount32(vandq_u32(w, a.0)),
                    vshlq_n_s32::<1>(popcount32(vandq_u32(w, a.1))),
                ),
                vshlq_n_s32::<2>(popcount32(vandq_u32(w, a.2))),
            ),
            vshlq_n_s32::<3>(popcount32(vandq_u32(w, a.3))),
        );
        let centered = vsubq_s32(vshlq_n_s32::<1>(dot), vld1q_s32(a_sums.add(g * 4)));
        let terms = vmulq_f32(vcvtq_f32_s32(centered), vld1q_f32(a_scales.add(g * 4)));
        total = vaddq_f32(total, vmulq_n_f32(terms, *scales.add(g) * 0.5));
    }
    vaddvq_f32(total)
}

/// Three CNTs, followed by a single shared pair of widening reductions.
/// Each byte is at most 8*(1+2+4)=56, so the byte weighting cannot overflow.
#[inline(always)]
unsafe fn signmag_count(weights: *const u8, planes: *const u32, g: usize) -> int32x4_t {
    let a = vld4q_u32(planes.add(g * 16));
    let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(g * 16)));
    let matches = veorq_u32(w, a.3);
    let c0 = vcntq_u8(vreinterpretq_u8_u32(vandq_u32(matches, a.0)));
    let c1 = vcntq_u8(vreinterpretq_u8_u32(vandq_u32(matches, a.1)));
    let c2 = vcntq_u8(vreinterpretq_u8_u32(vandq_u32(matches, a.2)));
    let weighted = vaddq_u8(vaddq_u8(c0, vshlq_n_u8::<1>(c1)), vshlq_n_u8::<2>(c2));
    vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(weighted)))
}

/// Safety: same lengths as dot_g32, plus negative_sums contains groups*4 i32.
/// Reconstructs the reference selected-weight integer dot, preserving FP32 order.
#[inline(never)]
pub unsafe fn dot_signmag_exact_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    negative_sums: *const i32,
    groups: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for g in 0..groups {
        let dot = vsubq_s32(
            signmag_count(weights, planes, g),
            vld1q_s32(negative_sums.add(g * 4)),
        );
        let weighted = vmulq_n_f32(vcvtq_f32_s32(dot), *scales.add(g));
        let bias = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4))), *biases.add(g));
        let terms = vmulq_f32(vaddq_f32(weighted, bias), vld1q_f32(a_scales.add(g * 4)));
        sum += vgetq_lane_f32::<0>(terms);
        sum += vgetq_lane_f32::<1>(terms);
        sum += vgetq_lane_f32::<2>(terms);
        sum += vgetq_lane_f32::<3>(terms);
    }
    sum
}

#[inline(always)]
unsafe fn signmag_centered_terms(
    weights: *const u8,
    planes: *const u32,
    a_scales: *const f32,
    absolute_sums: *const i32,
    g: usize,
) -> float32x4_t {
    let dot = vsubq_s32(
        vshlq_n_s32::<1>(signmag_count(weights, planes, g)),
        vld1q_s32(absolute_sums.add(g * 4)),
    );
    vmulq_f32(vcvtq_f32_s32(dot), vld1q_f32(a_scales.add(g * 4)))
}

/// Safety: weights/planes/scales as dot_g32; absolute_sums groups*4 i32.
/// Requires bias == -scale/2. Two vector accumulators and explicit FMA.
#[inline(never)]
pub unsafe fn dot_signmag_fast_g32(
    weights: *const u8,
    scales: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    absolute_sums: *const i32,
    groups: usize,
) -> f32 {
    let mut even = vdupq_n_f32(0.0);
    let mut odd = vdupq_n_f32(0.0);
    let mut g = 0;
    while g + 1 < groups {
        even = vfmaq_n_f32(
            even,
            signmag_centered_terms(weights, planes, a_scales, absolute_sums, g),
            *scales.add(g) * 0.5,
        );
        odd = vfmaq_n_f32(
            odd,
            signmag_centered_terms(weights, planes, a_scales, absolute_sums, g + 1),
            *scales.add(g + 1) * 0.5,
        );
        g += 2;
    }
    if g < groups {
        even = vfmaq_n_f32(
            even,
            signmag_centered_terms(weights, planes, a_scales, absolute_sums, g),
            *scales.add(g) * 0.5,
        );
    }
    vaddvq_f32(vaddq_f32(even, odd))
}

/// Four output rows share one LD4 and activation metadata per input block.
/// Safety: weights/scales/biases contain four complete rows; activation lengths
/// are as dot_signmag_exact_g32; output contains four writable floats.
#[inline(never)]
pub unsafe fn dot_signmag_exact4_g32(
    weights: *const u8,
    scales: *const f32,
    biases: *const f32,
    planes: *const u32,
    a_scales: *const f32,
    a_sums: *const i32,
    negative_sums: *const i32,
    groups: usize,
    output: *mut f32,
) {
    // SIMD lanes represent output rows. Transpose the four g32 terms, then
    // perform the original four additions in order for every row in parallel.
    let mut total = vdupq_n_f32(0.0);
    for g in 0..groups {
        let a = vld4q_u32(planes.add(g * 16));
        let scale = vld1q_f32(a_scales.add(g * 4));
        let sum = vcvtq_f32_s32(vld1q_s32(a_sums.add(g * 4)));
        let negative = vld1q_s32(negative_sums.add(g * 4));
        macro_rules! row {
            ($o:expr) => {{
                let w = vreinterpretq_u32_u8(vld1q_u8(weights.add(($o * groups + g) * 16)));
                let matches = veorq_u32(w, a.3);
                let c0 = vcntq_u8(vreinterpretq_u8_u32(vandq_u32(matches, a.0)));
                let c1 = vcntq_u8(vreinterpretq_u8_u32(vandq_u32(matches, a.1)));
                let c2 = vcntq_u8(vreinterpretq_u8_u32(vandq_u32(matches, a.2)));
                let weighted = vaddq_u8(vaddq_u8(c0, vshlq_n_u8::<1>(c1)), vshlq_n_u8::<2>(c2));
                let count = vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(weighted)));
                let dot = vcvtq_f32_s32(vsubq_s32(count, negative));
                vmulq_f32(
                    vaddq_f32(
                        vmulq_n_f32(dot, *scales.add($o * groups + g)),
                        vmulq_n_f32(sum, *biases.add($o * groups + g)),
                    ),
                    scale,
                )
            }};
        }
        let (r0, r1, r2, r3) = (row!(0), row!(1), row!(2), row!(3));
        let lo01 = vtrn1q_f32(r0, r1);
        let lo23 = vtrn1q_f32(r2, r3);
        let hi01 = vtrn2q_f32(r0, r1);
        let hi23 = vtrn2q_f32(r2, r3);
        let c0 = vcombine_f32(vget_low_f32(lo01), vget_low_f32(lo23));
        let c1 = vcombine_f32(vget_low_f32(hi01), vget_low_f32(hi23));
        let c2 = vcombine_f32(vget_high_f32(lo01), vget_high_f32(lo23));
        let c3 = vcombine_f32(vget_high_f32(hi01), vget_high_f32(hi23));
        total = vaddq_f32(total, c0);
        total = vaddq_f32(total, c1);
        total = vaddq_f32(total, c2);
        total = vaddq_f32(total, c3);
    }
    vst1q_f32(output, total);
}
