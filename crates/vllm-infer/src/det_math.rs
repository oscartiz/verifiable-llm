//! Deterministic transcendentals: exp, sin, cos with fully specified
//! algorithms over IEEE-754 f64 arithmetic — no libm, so results are
//! bit-identical on every platform that implements IEEE 754 (all Rust
//! targets we care about). Accuracy ~1e-15 relative, far beyond what the
//! deterministic inference path needs.
//!
//! Basic f32/f64 +,-,*,/ and sqrt are exactly specified by IEEE 754 and
//! deterministic everywhere; it is only the *library* functions (exp, sin,
//! cos, tanh, …) whose implementations vary between platforms. These
//! replacements pin them.

/// e^x for f64, via range reduction x = k·ln2 + r with |r| ≤ ln2/2 and a
/// degree-13 Taylor evaluation of e^r in Horner form (fixed order).
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -745.0 {
        return 0.0;
    }
    const LN2: f64 = core::f64::consts::LN_2;
    const INV_LN2: f64 = core::f64::consts::LOG2_E;
    let k = (x * INV_LN2 + if x >= 0.0 { 0.5 } else { -0.5 }).trunc();
    let r = x - k * LN2;
    // Taylor coefficients 1/n! for n = 13..=2, evaluated by Horner.
    const C: [f64; 12] = [
        1.0 / 6227020800.0, // 1/13!
        1.0 / 479001600.0,
        1.0 / 39916800.0,
        1.0 / 3628800.0,
        1.0 / 362880.0,
        1.0 / 40320.0,
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        1.0 / 2.0,
    ];
    let mut p = C[0];
    for &c in &C[1..] {
        p = p * r + c;
    }
    let er = (p * r + 1.0) * r + 1.0;
    // Scale by 2^k exactly via exponent manipulation.
    let ki = k as i64;
    let scale = f64::from_bits(((1023 + ki) as u64) << 52);
    er * scale
}

/// sin(x) and cos(x) for f64 via Cody–Waite style reduction modulo π/2 and
/// fixed Taylor polynomials. Adequate for |x| up to ~1e6 (RoPE arguments
/// stay below position_max ≈ 4096).
fn sin_cos_reduced(x: f64) -> (f64, f64) {
    // Split pi/2 for exact-ish reduction at moderate magnitudes: the std
    // constants are the correctly rounded f64 head; PIO2_LO is the tail.
    const PIO2_HI: f64 = core::f64::consts::FRAC_PI_2;
    const PIO2_LO: f64 = 6.123_233_995_736_766e-17;
    const INV_PIO2: f64 = core::f64::consts::FRAC_2_PI;

    let k = (x * INV_PIO2 + if x >= 0.0 { 0.5 } else { -0.5 }).trunc();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    let quadrant = (k as i64).rem_euclid(4);

    // Taylor series on |r| <= pi/4, Horner order fixed.
    let r2 = r * r;
    const S: [f64; 6] = [
        -1.0 / 6227020800.0, // -1/13!
        1.0 / 39916800.0,
        -1.0 / 362880.0,
        1.0 / 5040.0,
        -1.0 / 120.0,
        1.0 / 6.0,
    ];
    let mut sp = S[0];
    for &c in &S[1..] {
        sp = sp * r2 + c;
    }
    let sin_r = r - sp * r2 * r;

    const CC: [f64; 6] = [
        1.0 / 479001600.0, // 1/12!
        -1.0 / 3628800.0,
        1.0 / 40320.0,
        -1.0 / 720.0,
        1.0 / 24.0,
        -1.0 / 2.0,
    ];
    let mut cp = CC[0];
    for &c in &CC[1..] {
        cp = cp * r2 + c;
    }
    let cos_r = 1.0 + cp * r2;

    match quadrant {
        0 => (sin_r, cos_r),
        1 => (cos_r, -sin_r),
        2 => (-sin_r, -cos_r),
        _ => (-cos_r, sin_r),
    }
}

pub fn sin(x: f64) -> f64 {
    sin_cos_reduced(x).0
}

pub fn cos(x: f64) -> f64 {
    sin_cos_reduced(x).1
}

/// SiLU (x·sigmoid(x)) on f32 through the deterministic exp.
pub fn silu(x: f32) -> f32 {
    let xd = x as f64;
    (xd / (1.0 + exp(-xd))) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std_closely() {
        for i in -2000..2000 {
            let x = i as f64 * 0.037;
            let ours = exp(x);
            let std = x.exp();
            let rel = ((ours - std) / std).abs();
            assert!(rel < 1e-13, "exp({x}): {ours} vs {std} (rel {rel})");
        }
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp(800.0), f64::INFINITY);
        assert!(exp(f64::NAN).is_nan());
    }

    #[test]
    fn sin_cos_match_std_closely() {
        for i in -30000..30000 {
            let x = i as f64 * 0.173; // covers RoPE argument range (|x| < 5200)
            let (s, c) = (sin(x), cos(x));
            assert!((s - x.sin()).abs() < 1e-11, "sin({x}): {s} vs {}", x.sin());
            assert!((c - x.cos()).abs() < 1e-11, "cos({x}): {c} vs {}", x.cos());
        }
    }

    #[test]
    fn silu_sane() {
        assert!((silu(0.0) - 0.0).abs() < 1e-9);
        assert!((silu(10.0) - 9.99954) < 1e-3);
        assert!(silu(-10.0).abs() < 1e-3);
    }
}
