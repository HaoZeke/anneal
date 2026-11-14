//! Exchanges of unlike atoms scored by the objective alone.
//!
//! Nothing here reads a potential, an element or a size: an exchange is
//! two coordinates trading places between the indices of two labels, and
//! its score is the energy the caller returns for the result.

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// Indices of the two most common labels, larger group first.
fn label_groups(species: &[u32]) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut labels: Vec<(u32, usize)> = Vec::new();
    for &label in species {
        match labels.iter_mut().find(|(seen, _)| *seen == label) {
            Some(entry) => entry.1 += 1,
            None => labels.push((label, 1)),
        }
    }
    if labels.len() < 2 {
        return None;
    }
    labels.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    let pick = |label: u32| -> Vec<usize> {
        species
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (*item == label).then_some(index))
            .collect()
    };
    Some((pick(labels[0].0), pick(labels[1].0)))
}

/// Every unlike pair of the two most common labels, neither of them fixed.
pub(super) fn unlike_pairs(species: &[u32], fixed: impl Fn(usize) -> bool) -> Vec<(usize, usize)> {
    let Some((group_a, group_b)) = label_groups(species) else {
        return Vec::new();
    };
    let mut pairs = Vec::with_capacity(group_a.len() * group_b.len());
    for &i in &group_a {
        for &j in &group_b {
            if !fixed(i) && !fixed(j) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

fn swap_points(y: &mut Array1<f64>, i: usize, j: usize) {
    for axis in 0..3 {
        y.swap(3 * i + axis, 3 * j + axis);
    }
}

/// First-improvement descent over unlike exchanges at fixed coordinates.
///
/// `pairs` are visited in an order drawn from `seed`. Each exchange is
/// scored by the energy of the unrelaxed result and kept as soon as it is
/// lower than the current energy; the visit then goes on from the next
/// pair. It stops after a full pass that keeps nothing, or after
/// `max_reads` energies. Returns the reordered state and its unrelaxed
/// energy when an exchange was kept, with the number of energies read.
/// `None` from `energy` ends the descent.
pub(super) fn exchange_descent<E>(
    x: ArrayView1<f64>,
    energy_at_x: f64,
    pairs: &[(usize, usize)],
    max_reads: usize,
    seed: u64,
    mut energy: E,
) -> (Option<(f64, Array1<f64>)>, usize)
where
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let mut reads = 0usize;
    if pairs.is_empty() || !energy_at_x.is_finite() {
        return (None, reads);
    }
    let mut pairs = pairs.to_vec();
    pairs.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    let tolerance = 1e-9 * (1.0 + energy_at_x.abs());
    let mut current = x.to_owned();
    let mut current_e = energy_at_x;
    let mut moved = false;
    let mut since = 0usize;
    let mut at = 0usize;
    while since < pairs.len() && reads < max_reads {
        let (i, j) = pairs[at];
        at = (at + 1) % pairs.len();
        since += 1;
        swap_points(&mut current, i, j);
        reads += 1;
        let Some(value) = energy(current.view()) else {
            swap_points(&mut current, i, j);
            break;
        };
        if value.is_finite() && value < current_e - tolerance {
            current_e = value;
            moved = true;
            since = 0;
        } else {
            swap_points(&mut current, i, j);
        }
    }
    (moved.then_some((current_e, current)), reads)
}

/// The `keep` of `pairs` whose exchanged, unrelaxed energies are lowest.
///
/// Every pair is read once. `None` from `energy` ends the scan, and the
/// ranking covers the pairs read until then.
pub(super) fn ranked_exchanges<E>(
    x: ArrayView1<f64>,
    pairs: &[(usize, usize)],
    keep: usize,
    mut energy: E,
) -> Vec<(usize, usize)>
where
    E: FnMut(ArrayView1<f64>) -> Option<f64>,
{
    let mut probe = x.to_owned();
    let mut scored = Vec::new();
    for &(i, j) in pairs {
        swap_points(&mut probe, i, j);
        let value = energy(probe.view());
        swap_points(&mut probe, i, j);
        let Some(value) = value else {
            break;
        };
        scored.push((
            if value.is_finite() {
                value
            } else {
                f64::INFINITY
            },
            i,
            j,
        ));
    }
    scored.sort_by(|left, right| left.0.total_cmp(&right.0));
    scored
        .into_iter()
        .take(keep)
        .map(|(_, i, j)| (i, j))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIES: [u32; 4] = [0, 0, 1, 1];

    // Label 0 prefers small x and label 1 large x.
    fn line_energy(x: ArrayView1<f64>) -> Option<f64> {
        let mut total = 0.0;
        for (index, label) in SPECIES.iter().enumerate() {
            let position = x[3 * index];
            total += if *label == 0 { position } else { -position };
        }
        Some(total)
    }

    fn pairs() -> Vec<(usize, usize)> {
        unlike_pairs(&SPECIES, |_| false)
    }

    fn misordered() -> Array1<f64> {
        Array1::from_vec(vec![
            3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
    }

    #[test]
    fn a_misordered_line_is_sorted_by_exchanges() {
        let x = misordered();
        let start = line_energy(x.view()).unwrap();
        let (found, reads) = exchange_descent(x.view(), start, &pairs(), 64, 7, line_energy);
        let (value, y) = found.expect("an exchange lowers the energy");
        assert!(value < start);
        assert!(reads <= 64);
        assert!(y[0] < 2.0 && y[3] < 2.0, "label 0 ends at small x: {y:?}");
        let mut sorted: Vec<f64> = y.iter().copied().collect();
        let mut original: Vec<f64> = x.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        original.sort_by(f64::total_cmp);
        assert_eq!(sorted, original, "coordinates are only permuted");
    }

    #[test]
    fn the_read_cap_bounds_the_descent() {
        let x = misordered();
        let start = line_energy(x.view()).unwrap();
        let (found, reads) = exchange_descent(x.view(), start, &pairs(), 1, 7, line_energy);
        assert_eq!(reads, 1);
        assert!(found.is_none_or(|(value, _)| value < start));
    }

    #[test]
    fn an_ordered_state_costs_one_pass() {
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ]);
        let start = line_energy(x.view()).unwrap();
        let (found, reads) = exchange_descent(x.view(), start, &pairs(), 64, 3, line_energy);
        assert!(found.is_none());
        assert_eq!(reads, 4);
    }

    #[test]
    fn one_label_has_nothing_to_exchange() {
        let x = Array1::zeros(6);
        let none = unlike_pairs(&[7, 7], |_| false);
        assert!(none.is_empty());
        let (found, reads) = exchange_descent(x.view(), 0.0, &none, 4, 1, |_| Some(0.0));
        assert!(found.is_none());
        assert_eq!(reads, 0);
        assert!(ranked_exchanges(x.view(), &none, 4, |_| Some(0.0)).is_empty());
    }

    #[test]
    fn a_fixed_atom_is_never_exchanged() {
        let kept = unlike_pairs(&SPECIES, |atom| atom == 2);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|&(i, j)| i != 2 && j != 2));
    }

    #[test]
    fn the_ranking_leads_with_the_lowest_exchange() {
        let x = misordered();
        let ranked = ranked_exchanges(x.view(), &pairs(), 2, line_energy);
        assert_eq!(ranked.len(), 2);
        assert_eq!(
            ranked[0],
            (0, 3),
            "the widest misplacement is exchanged first"
        );
    }
}
