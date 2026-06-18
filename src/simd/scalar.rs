//! Naive scalar dot product. Always compiled. Acts as the bit-exact
//! reference oracle for the parity tests (CLAUDE.md §8.3).
//!
//! The reduction is intentionally structured as 8 parallel lane
//! accumulators followed by a sequential left-fold. SIMD paths mirror this
//! shape after spilling their vector registers, which is what makes the
//! parity test pass without a tolerance.

/// Number of parallel lane accumulators in the reduction. Every SIMD kernel
/// in this crate must spill into 8 lanes before the final left-fold.
pub(crate) const LANES: usize = 8;

/// Naive `f32` dot product. Bit-exact reference.
///
/// Uses [`f32::mul_add`] so that hardware-FMA SIMD paths
/// (`_mm256_fmadd_ps`, `vfmaq_f32`) produce identical results — both
/// perform `a * b + c` with a single rounding step.
///
/// Panics on length mismatch.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "scalar::dot: length mismatch");
    let n = a.len();
    let main = n - n % LANES;

    let mut lanes = [0.0f32; LANES];
    let mut i = 0;
    while i < main {
        // Lane-grouped FMA. The compiler is free to auto-vectorize this — it
        // does not change the per-lane semantics, only the schedule.
        for k in 0..LANES {
            lanes[k] = a[i + k].mul_add(b[i + k], lanes[k]);
        }
        i += LANES;
    }

    // Sequential left-fold across lane accumulators. Mirrors what each SIMD
    // path does after spilling its register(s) to an 8-element scratch.
    let mut sum = lanes[0];
    for &v in &lanes[1..] {
        sum += v;
    }

    // Scalar tail, also single-rounding FMA so SIMD tails match.
    while i < n {
        sum = a[i].mul_add(b[i], sum);
        i += 1;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn one_element() {
        assert_eq!(dot(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    fn full_chunk_no_tail() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        // 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8 = 120
        assert_eq!(dot(&a, &b), 120.0);
    }

    #[test]
    fn chunk_plus_tail() {
        let a: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..11).map(|i| (i as f32) * 2.0).collect();
        let expected: f32 = (0..11).map(|i| (i as f32) * (i as f32) * 2.0).sum();
        // Reference uses the same lane-grouped path, but for tiny inputs the
        // total is small enough that exact integer arithmetic agrees.
        assert_eq!(dot(&a, &b), expected);
    }

    #[test]
    #[should_panic]
    fn length_mismatch_panics() {
        dot(&[1.0, 2.0], &[1.0]);
    }
}
