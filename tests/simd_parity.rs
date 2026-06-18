//! Bit-exact parity tests — CLAUDE.md §8.3 and §12 (Milestone 1).
//!
//! Each enabled SIMD path must produce the exact same `f32` bit pattern as
//! [`scalar::dot`] across the listed dims. If a path drifts, the SIMD code
//! is wrong — *fix it, don't relax the test*.

use flats::simd::scalar;

/// Dimensions from CLAUDE.md §12 — cover pure-tail (1, 7), exact-multiple
/// (8, 16, 32, 128, 768, 1536), and one-past-boundary (9, 15, 31, 127).
const DIMS: &[usize] = &[1, 7, 8, 9, 15, 16, 31, 32, 127, 128, 768, 1536];

/// Deterministic PRNG so failures reproduce. SplitMix64-style; no deps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        // Linear congruential step → upper 24 bits → [-1, 1).
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32 & 0x00FF_FFFF; // 24-bit mantissa magnitude
        let mag = (bits as f32) / ((1u32 << 24) as f32); // [0, 1)
        let sign = if (self.0 >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        sign * mag
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn assert_bit_exact(label: &str, dim: usize, seed: u64, expected: f32, got: f32) {
    assert_eq!(
        expected.to_bits(),
        got.to_bits(),
        "{label}: dim={dim} seed={seed} scalar={expected:?} ({:#010x}) other={got:?} ({:#010x})",
        expected.to_bits(),
        got.to_bits()
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_parity_matches_scalar() {
    use flats::simd::avx2;

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("avx2/fma not available on this host; skipping");
        return;
    }
    for &dim in DIMS {
        for seed in 0..8u64 {
            let mut rng = Rng::new(seed * 0x1234_5678 + dim as u64);
            let a = rng.vec(dim);
            let b = rng.vec(dim);
            let r_ref = scalar::dot(&a, &b);
            // SAFETY: AVX2 + FMA confirmed by `is_x86_feature_detected!` above.
            let r_simd = unsafe { avx2::dot(&a, &b) };
            assert_bit_exact("avx2", dim, seed, r_ref, r_simd);
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_parity_matches_scalar() {
    use flats::simd::neon;

    if !std::arch::is_aarch64_feature_detected!("neon") {
        eprintln!("neon not available on this host; skipping");
        return;
    }
    for &dim in DIMS {
        for seed in 0..8u64 {
            let mut rng = Rng::new(seed * 0x1234_5678 + dim as u64);
            let a = rng.vec(dim);
            let b = rng.vec(dim);
            let r_ref = scalar::dot(&a, &b);
            // SAFETY: NEON confirmed by `is_aarch64_feature_detected!` above
            // (and is in the baseline aarch64 ABI anyway).
            let r_simd = unsafe { neon::dot(&a, &b) };
            assert_bit_exact("neon", dim, seed, r_ref, r_simd);
        }
    }
}

#[test]
fn dispatcher_matches_scalar() {
    for &dim in DIMS {
        for seed in 0..8u64 {
            let mut rng = Rng::new(seed.wrapping_add(0xDEAD_BEEF) ^ dim as u64);
            let a = rng.vec(dim);
            let b = rng.vec(dim);
            let r_ref = scalar::dot(&a, &b);
            let r_disp = flats::dot(&a, &b);
            assert_bit_exact("dispatch", dim, seed, r_ref, r_disp);
        }
    }
}

#[test]
fn scalar_against_textbook_for_tiny_inputs() {
    // Sanity: with small integer-valued inputs we know the exact answer.
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    // 10 + 40 + 90 + 160 = 300
    assert_eq!(scalar::dot(&a, &b), 300.0);
    assert_eq!(flats::dot(&a, &b), 300.0);
}
