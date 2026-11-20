//! Small paths `scripts/coverage.sh` showed no test reaching.

use flats::Param;
use flats::simd::scalar;

/// `dot_relaxed` is the benchmark baseline (a lane-split sum, not bit-exact
/// with the reference), but it is still public and must compute a dot
/// product: within rounding of `scalar::dot`, at every length — the lane
/// remainder included — and it must refuse mismatched lengths.
#[test]
fn dot_relaxed_is_a_dot_product_at_every_length() {
    let mut x = 0x2545_F491u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x % 2001) as f32 - 1000.0) / 100.0
    };
    for len in (0..70).chain([127, 128, 129, 768, 1536]) {
        let a: Vec<f32> = (0..len).map(|_| next()).collect();
        let b: Vec<f32> = (0..len).map(|_| next()).collect();
        let exact = scalar::dot(&a, &b);
        let relaxed = scalar::dot_relaxed(&a, &b);
        let mag: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
        assert!(
            (exact - relaxed).abs() <= 1e-5 * mag + 1e-6,
            "len {len}: {relaxed} vs {exact}"
        );
    }
    let r = std::panic::catch_unwind(|| scalar::dot_relaxed(&[1.0, 2.0], &[1.0]));
    assert!(r.is_err(), "a length mismatch is refused");
}

/// `Param` converts from the obvious Rust types, so `execute_with` reads
/// naturally: `&[7.into(), "x".into(), vec.into()]`.
#[test]
fn params_convert_from_plain_rust_values() {
    let v = vec![1.0f32, 2.0];
    let ps: Vec<Param> = vec![
        7i64.into(),
        0.5f64.into(),
        "a".into(),
        String::from("b").into(),
        v.clone().into(),
        v.as_slice().into(),
    ];
    assert_eq!(
        format!("{ps:?}"),
        format!(
            "{:?}",
            vec![
                Param::Int(7),
                Param::Float(0.5),
                Param::Text("a".into()),
                Param::Text("b".into()),
                Param::Vector(v.clone()),
                Param::Vector(v),
            ]
        )
    );
}
