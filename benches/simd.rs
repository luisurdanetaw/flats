//! SIMD kernel microbench. Raw timing harness — no `criterion` dep
//! (CLAUDE.md §2, §11). Run with:
//!
//! ```sh
//! cargo bench --bench simd
//! ```
//!
//! Reports ns/call and effective GB/s (bytes read = 2 * dim * 4) per kernel.
//! The CI gate for Milestone 1 is "AVX2 ≥ 4× scalar @ dim=768" — eyeball it
//! in the printed numbers; auto-checking lives outside this harness.

use std::hint::black_box;
use std::time::{Duration, Instant};

use flats::simd::scalar;

const TARGET: Duration = Duration::from_millis(250);
const DIMS: &[usize] = &[128, 768, 1536, 4096];

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32 & 0x00FF_FFFF;
        let mag = (bits as f32) / ((1u32 << 24) as f32);
        let sign = if (self.0 >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        sign * mag
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn run<F: Fn(&[f32], &[f32]) -> f32>(name: &str, dim: usize, a: &[f32], b: &[f32], f: F) -> f64 {
    // Pilot timing: double `iters` until we cross TARGET. The last pass is
    // the one we report — earlier passes serve as warmup.
    let mut iters: u64 = 1024;
    loop {
        let start = Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..iters {
            // black_box on inputs prevents the compiler from hoisting the
            // dot product out of the loop.
            let r = f(black_box(a), black_box(b));
            sink += black_box(r);
        }
        let elapsed = start.elapsed();
        black_box(sink);

        if elapsed >= TARGET || iters >= 1_000_000_000 {
            let ns_per = elapsed.as_nanos() as f64 / iters as f64;
            let bytes_per_call = 8.0 * dim as f64;
            let gbps = bytes_per_call / ns_per; // bytes/ns == GB/s
            println!(
                "  {:<14} dim={:<5} {:>10.2} ns/call  {:>7.2} GB/s  ({} iters, {:.2?})",
                name, dim, ns_per, gbps, iters, elapsed
            );
            return ns_per;
        }
        iters = iters.saturating_mul(4);
    }
}

fn main() {
    println!(
        "flats SIMD bench — selected kernel: {:?}",
        flats::simd::selected_kernel()
    );
    println!();

    for &dim in DIMS {
        let mut rng_a = Rng::new(0xA5A5_A5A5);
        let mut rng_b = Rng::new(0x5A5A_5A5A);
        let a = rng_a.vec(dim);
        let b = rng_b.vec(dim);

        println!("dim = {dim}");
        let scalar_ns = run("scalar", dim, &a, &b, scalar::dot);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: features confirmed above.
                let simd_ns = run("avx2+fma", dim, &a, &b, |a, b| unsafe {
                    flats::simd::avx2::dot(a, b)
                });
                println!("    speedup: avx2/scalar = {:.2}x", scalar_ns / simd_ns);
            } else {
                println!("    avx2+fma: not available on this host");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: neon confirmed above (and baseline on aarch64).
                let simd_ns = run("neon", dim, &a, &b, |a, b| unsafe {
                    flats::simd::neon::dot(a, b)
                });
                println!("    speedup: neon/scalar = {:.2}x", scalar_ns / simd_ns);
            } else {
                println!("    neon: not available on this host");
            }
        }

        // Dispatched path — first call warms the cache; the bench loop then
        // measures the hot path including the atomic load + match.
        let _warm = flats::dot(&a, &b);
        black_box(_warm);
        run("dispatch", dim, &a, &b, flats::dot);
        println!();
    }
}
