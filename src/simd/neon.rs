//! NEON dot product (aarch64).
//!
//! NEON has 128-bit registers (4 `f32` lanes). We carry two accumulators so
//! every main-loop iteration absorbs 8 elements, matching the scalar
//! reference's 8-lane reduction shape. `vfmaq_f32` is a single-rounding FMA,
//! matching `f32::mul_add` and `_mm256_fmadd_ps` bit-for-bit.

use core::arch::aarch64::*;

const LANES: usize = 8;

/// NEON dot product.
///
/// # Safety
///
/// The caller must guarantee NEON is available. NEON is in the baseline
/// aarch64 ABI, but we still gate on the dispatcher's runtime detection
/// for symmetry with the x86_64 path.
#[target_feature(enable = "neon")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "neon::dot: length mismatch");
    let n = a.len();
    let main = n - n % LANES;

    // SAFETY: `#[target_feature(enable = "neon")]` plus the dispatcher's
    // runtime detection (NEON is baseline on aarch64). `vld1q_f32` /
    // `vst1q_f32` accept any 4-byte-aligned pointer; `&[f32]` satisfies
    // that. Pointer arithmetic stays within bounds (`i + LANES <= main <= n`).
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0); // lanes 0..4
        let mut acc1 = vdupq_n_f32(0.0); // lanes 4..8
        let mut i = 0;
        while i < main {
            let a0 = vld1q_f32(a.as_ptr().add(i));
            let b0 = vld1q_f32(b.as_ptr().add(i));
            let a1 = vld1q_f32(a.as_ptr().add(i + 4));
            let b1 = vld1q_f32(b.as_ptr().add(i + 4));
            acc0 = vfmaq_f32(acc0, a0, b0);
            acc1 = vfmaq_f32(acc1, a1, b1);
            i += LANES;
        }

        // Spill into a contiguous 8-element scratch so the left-fold order
        // matches the AVX2 and scalar reductions exactly.
        let mut tmp = [0.0f32; LANES];
        vst1q_f32(tmp.as_mut_ptr(), acc0);
        vst1q_f32(tmp.as_mut_ptr().add(4), acc1);
        let mut sum = tmp[0];
        for &v in &tmp[1..] {
            sum += v;
        }

        // Scalar tail with single-rounding FMA.
        let mut j = i;
        while j < n {
            sum = a[j].mul_add(b[j], sum);
            j += 1;
        }
        sum
    }
}
