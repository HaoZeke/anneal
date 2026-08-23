//! Bond-orientational order, the continuous fivefold coordinate.
//!
//! The polyhedral-template shares separate the two LJ75 funnels at their
//! floors -- fivefold 0.307 icosahedral against 0.120 for the Marks
//! decahedron -- but the classifier is discrete: an atom either matches a
//! template or does not, so the share cannot drive a gradient and its
//! thresholded fires carried likelihood ratio one on the staging
//! structures. Steinhardt's \(q_6\) is the continuous version of the same
//! geometry: the second-moment invariant of the bond directions around
//! each atom, near 0.66 for an icosahedral cage and near 0.57 for the
//! close packing a decahedron is mostly made of, with no reference
//! structure anywhere in its definition.
//!
//! Steinhardt, Nelson, Ronchetti, *Phys. Rev. B* **1983**, *28*, 784.

use ndarray::ArrayView1;

/// Real spherical harmonics \(Y_{6m}\) squared-sum accumulator per centre.
///
/// The invariant needs \(\sum_m |q_{6m}|^2\), so the complex harmonics are
/// carried as their real and imaginary parts through the associated
/// Legendre column at \(l=6\).
fn legendre6(x: f64) -> [f64; 7] {
    // P_6^m(x) for m = 0..6, with Condon-Shortley phase folded into the
    // normalisation below rather than carried here.
    let s2 = (1.0 - x * x).max(0.0);
    let s = s2.sqrt();
    let x2 = x * x;
    [
        (231.0 * x2 * x2 * x2 - 315.0 * x2 * x2 + 105.0 * x2 - 5.0) / 16.0,
        21.0 / 8.0 * x * (33.0 * x2 * x2 - 30.0 * x2 + 5.0) * s,
        105.0 / 32.0 * (33.0 * x2 * x2 - 18.0 * x2 + 1.0) * s2,
        315.0 / 8.0 * x * (11.0 * x2 - 3.0) * s2 * s,
        945.0 / 16.0 * (11.0 * x2 - 1.0) * s2 * s2,
        10395.0 / 8.0 * x * s2 * s2 * s,
        10395.0 / 16.0 * s2 * s2 * s2,
    ]
}

/// \(\bar q_6\): the mean over atoms of the per-atom \(q_6\) invariant,
/// bonds counted inside `cutoff`.
///
/// Returns `None` when any atom has no neighbour, which is a broken
/// geometry rather than a phase.
pub fn mean_q6(x: ArrayView1<f64>, cutoff: f64) -> Option<f64> {
    let n = x.len() / 3;
    if n == 0 || x.len() % 3 != 0 {
        return None;
    }
    // Normalisations K_m = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!), l = 6.
    const L: usize = 6;
    let mut k = [0.0_f64; L + 1];
    let pref = (13.0 / (4.0 * std::f64::consts::PI)).sqrt();
    for (m, slot) in k.iter_mut().enumerate() {
        let mut ratio = 1.0_f64;
        for j in (L - m + 1)..=(L + m) {
            ratio *= j as f64;
        }
        *slot = pref / ratio.sqrt();
    }
    let at = |i: usize, d: usize| x[3 * i + d];
    let mut total = 0.0;
    for i in 0..n {
        // q_{6m} accumulators, m = 0..6, real and imaginary.
        let mut re = [0.0_f64; L + 1];
        let mut im = [0.0_f64; L + 1];
        let mut bonds = 0usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = at(j, 0) - at(i, 0);
            let dy = at(j, 1) - at(i, 1);
            let dz = at(j, 2) - at(i, 2);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > cutoff || r < 1e-12 {
                continue;
            }
            bonds += 1;
            let ct = dz / r;
            let phi = dy.atan2(dx);
            let p = legendre6(ct);
            for m in 0..=L {
                let ylm = k[m] * p[m];
                let (sphi, cphi) = (m as f64 * phi).sin_cos();
                re[m] += ylm * cphi;
                im[m] += ylm * sphi;
            }
        }
        if bonds == 0 {
            return None;
        }
        let nb = bonds as f64;
        // sum over m of |q_6m|^2, negative m folded in as a factor of two.
        let mut s = re[0] * re[0] / (nb * nb);
        for m in 1..=L {
            s += 2.0 * (re[m] * re[m] + im[m] * im[m]) / (nb * nb);
        }
        total += (4.0 * std::f64::consts::PI / 13.0 * s).sqrt();
    }
    Some(total / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn a_perfect_fcc_shell_carries_the_textbook_q6() {
        // Twelve FCC nearest neighbours around a centre: q6 = 0.5745
        // (Steinhardt, Nelson, Ronchetti 1983, Table I).
        let mut pts = vec![0.0, 0.0, 0.0];
        let s = 1.0 / (2.0_f64).sqrt();
        for (a, b) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            pts.extend_from_slice(&[a * s, b * s, 0.0]);
            pts.extend_from_slice(&[a * s, 0.0, b * s]);
            pts.extend_from_slice(&[0.0, a * s, b * s]);
        }
        let x = Array1::from(pts);
        // Only the centre has all twelve inside the cutoff; the invariant
        // is checked at the centre by using a cutoff that isolates it.
        let n = x.len() / 3;
        let q = per_centre_q6(x.view(), 0, 1.1).expect("centre has bonds");
        assert!(n == 13);
        assert!(
            (q - 0.5745).abs() < 2e-3,
            "fcc shell q6 must be 0.5745, got {q}"
        );
    }

    #[test]
    fn a_perfect_icosahedral_shell_carries_the_textbook_q6() {
        // Twelve icosahedral vertices: q6 = 0.6633.
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let norm = (1.0 + p * p).sqrt();
        let mut pts = vec![0.0, 0.0, 0.0];
        for (a, b) in [(1.0, p), (1.0, -p), (-1.0, p), (-1.0, -p)] {
            pts.extend_from_slice(&[0.0, a / norm, b / norm]);
            pts.extend_from_slice(&[a / norm, b / norm, 0.0]);
            pts.extend_from_slice(&[b / norm, 0.0, a / norm]);
        }
        let x = Array1::from(pts);
        let q = per_centre_q6(x.view(), 0, 1.1).expect("centre has bonds");
        assert!(
            (q - 0.6633).abs() < 2e-3,
            "icosahedral shell q6 must be 0.6633, got {q}"
        );
    }
}

/// Per-centre \(q_6\), the term [`mean_q6`] averages. Public so a probe
/// can look at one atom's cage.
pub fn per_centre_q6(x: ArrayView1<f64>, centre: usize, cutoff: f64) -> Option<f64> {
    let n = x.len() / 3;
    if centre >= n {
        return None;
    }
    const L: usize = 6;
    let mut k = [0.0_f64; L + 1];
    let pref = (13.0 / (4.0 * std::f64::consts::PI)).sqrt();
    for (m, slot) in k.iter_mut().enumerate() {
        let mut ratio = 1.0_f64;
        for j in (L - m + 1)..=(L + m) {
            ratio *= j as f64;
        }
        *slot = pref / ratio.sqrt();
    }
    let at = |i: usize, d: usize| x[3 * i + d];
    let mut re = [0.0_f64; L + 1];
    let mut im = [0.0_f64; L + 1];
    let mut bonds = 0usize;
    for j in 0..n {
        if centre == j {
            continue;
        }
        let dx = at(j, 0) - at(centre, 0);
        let dy = at(j, 1) - at(centre, 1);
        let dz = at(j, 2) - at(centre, 2);
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r > cutoff || r < 1e-12 {
            continue;
        }
        bonds += 1;
        let ct = dz / r;
        let phi = dy.atan2(dx);
        let p = legendre6(ct);
        for m in 0..=L {
            let ylm = k[m] * p[m];
            let (sphi, cphi) = (m as f64 * phi).sin_cos();
            re[m] += ylm * cphi;
            im[m] += ylm * sphi;
        }
    }
    if bonds == 0 {
        return None;
    }
    let nb = bonds as f64;
    let mut s = re[0] * re[0] / (nb * nb);
    for m in 1..=L {
        s += 2.0 * (re[m] * re[m] + im[m] * im[m]) / (nb * nb);
    }
    Some((4.0 * std::f64::consts::PI / 13.0 * s).sqrt())
}
