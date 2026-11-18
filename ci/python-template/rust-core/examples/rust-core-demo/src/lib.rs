// MIT License
// Copyright 2023--present rgpot developers

/// Fixed-order dot product. The sum walks indices 0..n, so the result
/// does not depend on a thread count. Panic is caught at the ABI edge.
///
/// # Safety
///
/// `a` and `b` must each be null or point at `n` readable `f64` values.
#[no_mangle]
pub unsafe extern "C" fn rust_core_demo_dot(a: *const f64, b: *const f64, n: usize) -> f64 {
    let caught = std::panic::catch_unwind(|| unsafe {
        if a.is_null() || b.is_null() {
            return f64::NAN;
        }
        let left = std::slice::from_raw_parts(a, n);
        let right = std::slice::from_raw_parts(b, n);
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += left[i] * right[i];
        }
        acc
    });
    // Panic at this ABI edge becomes NaN. unwrap_or is not used here.
    #[allow(
        clippy::manual_unwrap_or,
        reason = "panic at this ABI edge becomes NaN"
    )]
    if let Ok(value) = caught {
        value
    } else {
        f64::NAN
    }
}
