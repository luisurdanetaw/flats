//! AVX2 + FMA dot product (x86_64).
//!
//! Eight `f32` lanes per `__m256`. We use one accumulator register so the
//! per-iteration arithmetic chain is a single `vfmadd231ps`. Unaligned
//! loads (`_mm256_loadu_ps`) per CLAUDE.md §8.4 — mmap'd vectors are only
//! guaranteed 4-byte aligned.

use core::arch::x86_64::*;

const LANES: usize = 8;

/// AVX2 + FMA dot product.
///
/// # Safety
///
/// The caller must guarantee that the CPU supports both `avx2` and `fma`.
/// The crate dispatcher in [`super`] enforces this via
/// `is_x86_feature_detected!`. Calling this function on a CPU without those
/// features is undefined behavior.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "avx2::dot: length mismatch");
    let n = a.len();
    let main = n - n % LANES;

    // SAFETY: `#[target_feature(enable = "avx2,fma")]` plus the dispatcher's
    // runtime detection guarantee the required CPU features are present.
    // `_mm256_loadu_ps` / `_mm256_storeu_ps` accept any pointer with at least
    // 1-byte alignment; `&[f32]` always satisfies that. The pointer
    // arithmetic `a.as_ptr().add(i)` stays within bounds (`i + LANES <= main
    // <= n`) so it never escapes the underlying allocation.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < main {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(av, bv, acc);
            i += LANES;
        }

        // Spill the 8 lanes to scratch and sequentially left-fold them,
        // matching scalar.rs and neon.rs reduction order bit-for-bit.
        let mut tmp = [0.0f32; LANES];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
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
