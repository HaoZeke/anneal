//! The length scale of a structure, read from the structure.
//!
//! Search code that needs a bond length takes it from the coordinates it is
//! standing on, so it carries no knowledge of the potential. This module holds
//! that one measurement. It builds no structure: candidates are never grown
//! from an order, a template or a lattice here.

use ndarray::ArrayView1;

/// Mean distance to the nearest other point.
///
/// The length scale a move is taken at, read from the structure so the move
/// carries no knowledge of the potential.
pub fn nearest_neighbour_scale(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    if n < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum();
            best = best.min(d);
        }
        total += best.sqrt();
    }
    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn the_scale_follows_the_structure_not_a_constant() {
        let line = |spacing: f64| {
            let mut x = Array1::zeros(12);
            for i in 0..4 {
                x[3 * i] = spacing * i as f64;
            }
            x
        };
        let narrow = nearest_neighbour_scale(line(1.0).view());
        let wide = nearest_neighbour_scale(line(2.5).view());
        assert!((narrow - 1.0).abs() < 1e-12, "{narrow}");
        assert!((wide - 2.5).abs() < 1e-12, "{wide}");
    }

    #[test]
    fn a_single_point_has_unit_scale() {
        assert_eq!(nearest_neighbour_scale(Array1::zeros(3).view()), 1.0);
    }
}
