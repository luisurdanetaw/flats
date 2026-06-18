//! Dot-product kernels with cached runtime dispatch.
//!
//! The public entry point [`dot`] auto-selects the fastest implementation
//! available on the host CPU on first call and caches the choice in an
//! `AtomicU8` (CLAUDE.md §8.2). No build-time feature flags: one binary
//! runs on every supported CPU.
//!
//! ## Bit-exact correctness
//!
//! All SIMD paths must agree with [`scalar::dot`] bit-for-bit (CLAUDE.md §8.3).
//! That is achieved by:
//!
//! 1. Every kernel processes the input in 8-element chunks and keeps 8
//!    parallel lane accumulators, regardless of native register width.
//! 2. Every kernel reduces by spilling the 8 lanes to a scratch array and
//!    summing them with a sequential left-fold (`tmp[0] + tmp[1] + … + tmp[7]`).
//! 3. Every kernel uses single-rounding FMA: [`f32::mul_add`] in scalar,
//!    `_mm256_fmadd_ps` on x86_64, `vfmaq_f32` on aarch64.
//! 4. The scalar tail is identical across kernels.

use core::sync::atomic::{AtomicU8, Ordering};

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

const KIND_UNINIT: u8 = 0;
const KIND_SCALAR: u8 = 1;
#[cfg(target_arch = "x86_64")]
const KIND_AVX2: u8 = 2;
#[cfg(target_arch = "aarch64")]
const KIND_NEON: u8 = 3;

static KIND: AtomicU8 = AtomicU8::new(KIND_UNINIT);

fn detect() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return KIND_AVX2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return KIND_NEON;
        }
    }
    KIND_SCALAR
}

fn cached_kind() -> u8 {
    let v = KIND.load(Ordering::Relaxed);
    if v != KIND_UNINIT {
        return v;
    }
    // A benign race here is fine: every thread computes the same answer and
    // the store is idempotent. `Relaxed` is sufficient since the byte carries
    // no other data.
    let d = detect();
    KIND.store(d, Ordering::Relaxed);
    d
}

/// Dot product of two `f32` slices.
///
/// Selects the fastest available kernel on first call and caches the choice.
/// Panics if the slices have different lengths.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    match cached_kind() {
        #[cfg(target_arch = "x86_64")]
        KIND_AVX2 => {
            // SAFETY: `cached_kind` only returns `KIND_AVX2` after
            // `is_x86_feature_detected!` confirmed both `avx2` and `fma`.
            unsafe { avx2::dot(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        KIND_NEON => {
            // SAFETY: `cached_kind` only returns `KIND_NEON` after
            // `is_aarch64_feature_detected!` confirmed `neon`.
            unsafe { neon::dot(a, b) }
        }
        _ => scalar::dot(a, b),
    }
}

/// Identifier of the kernel that the dispatcher will use on this host.
///
/// Resolves and caches the choice on first call. Useful for tests, benches,
/// and diagnostics.
pub fn selected_kernel() -> Kernel {
    match cached_kind() {
        #[cfg(target_arch = "x86_64")]
        KIND_AVX2 => Kernel::Avx2Fma,
        #[cfg(target_arch = "aarch64")]
        KIND_NEON => Kernel::Neon,
        _ => Kernel::Scalar,
    }
}

/// The kernel selected by the runtime dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Portable scalar reference (also the bit-exact oracle).
    Scalar,
    /// x86_64 AVX2 + FMA path.
    Avx2Fma,
    /// aarch64 NEON path.
    Neon,
}
