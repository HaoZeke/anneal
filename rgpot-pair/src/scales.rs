//! Characteristic length and energy of a cluster potential.
//!
//! The length is the nearest-neighbor distance. For a pair potential
//! that is the minimum of the pair. For Sutton-Chen it is the
//! nearest-neighbor distance of the static fcc crystal. The energy is
//! the depth of that well, positive. Sutton-Chen's well is the
//! cohesive energy per atom.

/// FCC lattice sum of `(r_nn / r)^power` through cubic index 32.
///
/// Sites are the fcc translations `(i, j, k) * a / 2` with `i + j + k`
/// even. `r_nn = a / sqrt(2)`. All even or all odd would be bcc.
pub fn fcc_power_sum(power: i32) -> f64 {
    const LIMIT: i32 = 32;
    let mut sum = 0.0;
    let p = power as f64;
    for i in -LIMIT..=LIMIT {
        for j in -LIMIT..=LIMIT {
            for k in -LIMIT..=LIMIT {
                if (i + j + k) & 1 != 0 {
                    continue;
                }
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }
                let r2 = f64::from(i * i + j * j + k * k);
                let inv = (2.0 / r2).sqrt();
                sum += inv.powf(p);
            }
        }
    }
    sum
}

/// Nearest-neighbor distance `sigma * 2^(1/6)` and well depth `epsilon`.
#[inline]
pub fn lennard_jones_scales(epsilon: f64, sigma: f64) -> (f64, f64) {
    (sigma * 2.0_f64.powf(1.0 / 6.0), epsilon)
}

/// Morse minimum sits at `re` with depth `epsilon`. `rho` does not move it.
#[inline]
pub fn morse_scales(epsilon: f64, _rho: f64, re: f64) -> (f64, f64) {
    (re, epsilon)
}

/// Parameters of one Sutton-Chen metal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuttonChenMetal {
    /// Energy unit, eV in the physical presets.
    pub epsilon: f64,
    /// Dimensionless density coefficient.
    pub c: f64,
    /// Lattice parameter, Angstrom in the physical presets.
    pub a: f64,
    /// Repulsive exponent.
    pub n: i32,
    /// Density exponent.
    pub m: i32,
}

/// Platinum 10-8 row. `epsilon` is `1.98330 * 10^-2` eV and `a` is 3.92 Angstrom.
pub fn sutton_chen_platinum() -> SuttonChenMetal {
    SuttonChenMetal {
        epsilon: 1.98330e-2,
        c: 34.408,
        a: 3.92,
        n: 10,
        m: 8,
    }
}

/// Gold 10-8 row. `epsilon` is `1.27930 * 10^-2` eV and `a` is 4.08 Angstrom.
pub fn sutton_chen_gold() -> SuttonChenMetal {
    SuttonChenMetal {
        epsilon: 1.27930e-2,
        c: 34.408,
        a: 4.08,
        n: 10,
        m: 8,
    }
}

/// Energy per atom of the static fcc crystal at nearest-neighbor distance `radius`.
///
/// `a` is the potential's lattice parameter. `radius` is the
/// nearest-neighbor spacing in the same unit.
pub fn sutton_chen_fcc_energy(epsilon: f64, c: f64, a: f64, n: i32, m: i32, radius: f64) -> f64 {
    let s_n = fcc_power_sum(n);
    let s_m = fcc_power_sum(m);
    let a_over_r = a / radius;
    let repulsive = 0.5 * a_over_r.powi(n) * s_n;
    let density = a_over_r.powf(0.5 * f64::from(m)) * s_m.sqrt();
    epsilon * (repulsive - c * density)
}

/// Bulk nearest-neighbor distance and cohesive well depth (positive).
pub fn sutton_chen_scales(epsilon: f64, c: f64, a: f64, n: i32, m: i32) -> (f64, f64) {
    let s_n = fcc_power_sum(n);
    let s_m = fcc_power_sum(m);
    let exponent = f64::from(n) - 0.5 * f64::from(m);
    let rhs = (c * f64::from(m) * s_m.sqrt()) / (f64::from(n) * s_n);
    let radius = a / rhs.powf(1.0 / exponent);
    let energy = sutton_chen_fcc_energy(epsilon, c, a, n, m, radius);
    (radius, -energy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lennard_jones_minimum_is_the_unit_well() {
        let (radius, depth) = lennard_jones_scales(1.0, 1.0);
        assert_eq!(depth, 1.0);
        let (energy, coef, _) = crate::lj_pair(1.0, 1.0, radius * radius);
        assert!((energy + 1.0).abs() < 1e-12);
        assert!(coef.abs() < 1e-10);
    }

    /// Conventional cubic cell and its four-site basis within a sphere
    /// of 32 cells. Independent of `fcc_power_sum`, which walks a cube
    /// in half-cell indices.
    fn direct_fcc_energy(m: &SuttonChenMetal, radius: f64) -> f64 {
        const CELLS: i32 = 32;
        let basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        let cell = radius * 2.0_f64.sqrt();
        let (mut pair, mut rho) = (0.0, 0.0);
        for i in -CELLS..=CELLS {
            for j in -CELLS..=CELLS {
                for k in -CELLS..=CELLS {
                    for b in &basis {
                        let x = f64::from(i) + b[0];
                        let y = f64::from(j) + b[1];
                        let z = f64::from(k) + b[2];
                        let d2 = x * x + y * y + z * z;
                        if d2 == 0.0 || d2 > f64::from(CELLS * CELLS) {
                            continue;
                        }
                        let s = m.a / (cell * d2.sqrt());
                        pair += s.powi(m.n);
                        rho += s.powi(m.m);
                    }
                }
            }
        }
        m.epsilon * (0.5 * pair - m.c * rho.sqrt())
    }

    #[test]
    fn bulk_scales_match_a_direct_fcc_sum() {
        let reduced = SuttonChenMetal {
            epsilon: 1.0,
            c: 34.408,
            a: 1.0,
            n: 10,
            m: 8,
        };
        for m in [reduced, sutton_chen_platinum(), sutton_chen_gold()] {
            let (radius, depth) = sutton_chen_scales(m.epsilon, m.c, m.a, m.n, m.m);
            assert!((radius / m.a - 0.5_f64.sqrt()).abs() < 1e-5, "{radius}");
            let here = direct_fcc_energy(&m, radius);
            assert!((here + depth).abs() / m.epsilon < 1e-4, "{here} {depth}");
            for step in [1e-3, -1e-3, 2e-2, -2e-2] {
                assert!(direct_fcc_energy(&m, radius * (1.0 + step)) > here);
            }
        }
        let unit = direct_fcc_energy(&reduced, 0.5_f64.sqrt());
        assert!((unit + 295.4645).abs() < 1e-3, "{unit}");
    }

    #[test]
    fn reduced_10_8_neighbor_is_stationary() {
        let (radius, depth) = sutton_chen_scales(1.0, 34.408, 1.0, 10, 8);
        let here = sutton_chen_fcc_energy(1.0, 34.408, 1.0, 10, 8, radius);
        let left = sutton_chen_fcc_energy(1.0, 34.408, 1.0, 10, 8, radius * (1.0 - 1e-4));
        let right = sutton_chen_fcc_energy(1.0, 34.408, 1.0, 10, 8, radius * (1.0 + 1e-4));
        assert!(here < left && here < right, "{left} {here} {right}");
        assert!((here + depth).abs() < 1e-9);
    }

    #[test]
    fn gold_and_platinum_rows() {
        let pt = sutton_chen_platinum();
        let au = sutton_chen_gold();
        assert_eq!(pt.epsilon, 1.98330e-2);
        assert_eq!(pt.a, 3.92);
        assert_eq!(pt.c, 34.408);
        assert_eq!(pt.n, 10);
        assert_eq!(pt.m, 8);
        assert_eq!(au.epsilon, 1.27930e-2);
        assert_eq!(au.a, 4.08);
        assert_eq!(au.c, 34.408);
        assert_eq!(au.n, 10);
        assert_eq!(au.m, 8);
        let (pt_r, _) = sutton_chen_scales(pt.epsilon, pt.c, pt.a, pt.n, pt.m);
        let (au_r, _) = sutton_chen_scales(au.epsilon, au.c, au.a, au.n, au.m);
        let ratio = au_r / pt_r;
        assert!((ratio - au.a / pt.a).abs() < 1e-12);
    }

    #[test]
    fn copper_silver_gold_and_nickel_match_published_bulk_values() {
        // Sutton and Chen, Phil. Mag. Lett. 61, 139 (1990), Table I.
        // The tabulated `a` is the conventional fcc lattice constant
        // and the cohesive energy is positive here.
        let rows = [
            (
                "Cu",
                SuttonChenMetal {
                    epsilon: 1.23820e-2,
                    c: 39.432,
                    a: 3.61,
                    n: 9,
                    m: 6,
                },
                3.50,
            ),
            (
                "Ag",
                SuttonChenMetal {
                    epsilon: 2.54150e-3,
                    c: 144.410,
                    a: 4.09,
                    n: 12,
                    m: 6,
                },
                2.96,
            ),
            ("Au", sutton_chen_gold(), 3.78),
            (
                "Ni",
                SuttonChenMetal {
                    epsilon: 1.57070e-2,
                    c: 39.432,
                    a: 3.52,
                    n: 9,
                    m: 6,
                },
                4.44,
            ),
        ];
        for (name, metal, published_depth) in rows {
            let (radius, depth) =
                sutton_chen_scales(metal.epsilon, metal.c, metal.a, metal.n, metal.m);
            let lattice_constant = radius * 2.0_f64.sqrt();
            assert!(
                (lattice_constant - metal.a).abs() < 1e-5,
                "{name}: a={lattice_constant}, expected {}",
                metal.a
            );
            assert!(
                (depth - published_depth).abs() < 2e-4,
                "{name}: cohesive energy={depth}, expected {published_depth}"
            );

            // This sum has a spherical cutoff and a four-site conventional
            // basis, independently of the parity walk in fcc_power_sum.
            let brute = direct_fcc_energy(&metal, radius);
            assert!(
                (brute + published_depth).abs() < 2e-4,
                "{name}: brute-force energy={brute}, expected -{published_depth}"
            );
            for strain in [-2e-3, 2e-3] {
                assert!(
                    direct_fcc_energy(&metal, radius * (1.0 + strain)) > brute,
                    "{name}: the published lattice is not a local minimum"
                );
            }
        }
    }
}
