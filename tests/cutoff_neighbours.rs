//! Cutoff neighbour lists: linkcell pairs inside a periodic cell, and
//! unwrapped pairs on an open axis.

use anneal_core::descriptor_space::{DescriptorGeometry, descriptor_cutoff_neighbours};
use anneal_core::methods::csa_cluster::coordination_shell_counts;
use anneal_core::neighbors::{CutoffNeighbour, cutoff_pairs, periodic_cutoff_pairs};
use linkcell::Cell;
use ndarray::Array1;

fn length2(value: [f64; 3]) -> f64 {
    value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
}

fn prefers_negative(candidate: [f64; 3], incumbent: [f64; 3]) -> bool {
    for axis in 0..3 {
        let delta = candidate[axis] - incumbent[axis];
        if delta < -1e-12 {
            return true;
        }
        if delta > 1e-12 {
            return false;
        }
    }
    false
}

/// Shortest lattice image of each ordered pair, searched directly.
/// An open axis contributes only the zero shift.
fn brute_shortest(
    positions: &[[f64; 3]],
    cell: &Cell,
    periodic: [bool; 3],
    cutoff: f64,
) -> Vec<Vec<CutoffNeighbour>> {
    let n = positions.len();
    let mut reach = [0_i32; 3];
    let widths = cell.widths();
    for axis in 0..3 {
        if periodic[axis] {
            let width = widths[axis].abs().max(1e-12);
            reach[axis] = ((cutoff / width).ceil() as i32 + 2).clamp(2, 8);
        }
    }
    let cutoff2 = cutoff * cutoff;
    let mut lists = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            let mut chosen: Option<(f64, [f64; 3])> = None;
            for na in -reach[0]..=reach[0] {
                for nb in -reach[1]..=reach[1] {
                    for nc in -reach[2]..=reach[2] {
                        if i == j && na == 0 && nb == 0 && nc == 0 {
                            continue;
                        }
                        let shift = cell.lattice_shift(na, nb, nc);
                        let p = positions[i];
                        let q = positions[j];
                        let displacement = [
                            q[0] + shift[0] - p[0],
                            q[1] + shift[1] - p[1],
                            q[2] + shift[2] - p[2],
                        ];
                        let d2 = length2(displacement);
                        if !(d2 < cutoff2) || d2 <= 1e-24 {
                            continue;
                        }
                        let replace = match chosen {
                            None => true,
                            Some((old2, old)) => {
                                if d2 < old2 - 1e-12 {
                                    true
                                } else if (old2 - d2).abs() <= 1e-9 * old2.max(1.0) {
                                    prefers_negative(displacement, old)
                                } else {
                                    false
                                }
                            }
                        };
                        if replace {
                            chosen = Some((d2, displacement));
                        }
                    }
                }
            }
            if let Some((_, displacement)) = chosen {
                lists[i].push(CutoffNeighbour {
                    index: j,
                    displacement,
                    distance: length2(displacement).sqrt(),
                });
            }
        }
        lists[i].sort_by_key(|neighbour| neighbour.index);
    }
    lists
}

fn assert_lists_match(got: &[Vec<CutoffNeighbour>], expect: &[Vec<CutoffNeighbour>]) {
    assert_eq!(got.len(), expect.len());
    for (centre, (left, right)) in got.iter().zip(expect.iter()).enumerate() {
        assert_eq!(left.len(), right.len(), "centre {centre} pair count");
        for (found, wanted) in left.iter().zip(right.iter()) {
            assert_eq!(found.index, wanted.index, "centre {centre}");
            for axis in 0..3 {
                assert!(
                    (found.displacement[axis] - wanted.displacement[axis]).abs() < 1e-9,
                    "centre {centre} atom {} axis {axis}: got {:?} want {:?}",
                    found.index,
                    found.displacement,
                    wanted.displacement
                );
            }
        }
    }
}

#[test]
fn periodic_cutoff_matches_the_shortest_lattice_vector() {
    let ortho = Cell::ortho(10.0, 12.0, 14.0).expect("orthorhombic cell");
    let ortho_points = [
        [0.0, 1.0, 2.0],
        [5.0, 1.0, 2.0],
        [1.2, 11.4, 2.5],
        [9.7, 0.4, 13.2],
    ];
    let ortho_cutoff = 6.0;
    let got = periodic_cutoff_pairs(&ortho_points, &ortho, ortho_cutoff).expect("ortho pairs");
    let expect = brute_shortest(&ortho_points, &ortho, [true; 3], ortho_cutoff);
    assert_lists_match(&got, &expect);
    let half = got[0]
        .iter()
        .find(|neighbour| neighbour.index == 1)
        .expect("atom 0 sees the half-box atom");
    assert!(
        (half.displacement[0] + 5.0).abs() < 1e-9,
        "the half-box tie keeps -L/2, got {:?}",
        half.displacement
    );
    assert!((half.displacement[1]).abs() < 1e-9);
    assert!((half.displacement[2]).abs() < 1e-9);

    let open_vectors = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]];
    let open_points = [[0.2, 0.0, 0.0], [9.6, 0.3, 8.0], [1.0, 1.0, 1.0]];
    let open_cutoff = 9.0;
    let open_cell = Cell::from_vectors(open_vectors[0], open_vectors[1], open_vectors[2], [0.0; 3])
        .expect("open-axis cell");
    let got = cutoff_pairs(&open_points, open_vectors, [true, true, false], open_cutoff)
        .expect("open-axis pairs");
    let expect = brute_shortest(&open_points, &open_cell, [true, true, false], open_cutoff);
    assert_lists_match(&got, &expect);
    let across = got[0]
        .iter()
        .find(|neighbour| neighbour.index == 1)
        .expect("atom 0 sees the point across the open axis");
    assert!(
        (across.displacement[2] - 8.0).abs() < 1e-9,
        "the open axis is not wrapped, got {:?}",
        across.displacement
    );

    let skew_vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]];
    let skew_points = [[0.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.1, 0.02, 0.4]];
    let skew_cutoff = 0.5;
    let skew = Cell::from_vectors(skew_vectors[0], skew_vectors[1], skew_vectors[2], [0.0; 3])
        .expect("skewed cell");
    let got = periodic_cutoff_pairs(&skew_points, &skew, skew_cutoff).expect("skewed pairs");
    let expect = brute_shortest(&skew_points, &skew, [true; 3], skew_cutoff);
    assert_lists_match(&got, &expect);

    // Hexagonal prism. The fractional wrap of the body diagonal is longer
    // than another lattice image, so the shipped vector has to be that
    // shorter image.
    let hex_a = [10.0, 0.0, 0.0];
    let hex_b = [5.0, 8.660254037844386, 0.0];
    let hex_c = [0.0, 0.0, 10.0];
    let hex = Cell::from_vectors(hex_a, hex_b, hex_c, [0.0; 3]).expect("hexagonal cell");
    let hex_points = [hex.cartesian([0.49, 0.49, 0.49]), [0.0; 3]];
    let hex_cutoff = 9.85;
    let got = periodic_cutoff_pairs(&hex_points, &hex, hex_cutoff).expect("hex pairs");
    let expect = brute_shortest(&hex_points, &hex, [true; 3], hex_cutoff);
    assert_lists_match(&got, &expect);
    let named = got[0]
        .iter()
        .find(|neighbour| neighbour.index == 1)
        .expect("pair (0, 1) is inside the cutoff");
    let fractional = hex.displacement(hex_points[0], hex_points[1]);
    assert!(
        length2(fractional) > length2(named.displacement) + 1e-8,
        "pair (0, 1) fractional wrap {:?} is longer than the shortest {:?}",
        fractional,
        named.displacement
    );
}

#[test]
fn free_cluster_shells_match_cartesian_pairs() {
    let points = [
        [0.0, 0.0, 0.0],
        [0.7, 0.1, 0.0],
        [1.4, 0.2, 0.1],
        [0.2, 1.3, 0.4],
        [0.4, 0.5, 1.6],
    ];
    let n = points.len();
    let mut flat = Array1::zeros(3 * n);
    for (atom, point) in points.iter().enumerate() {
        flat[3 * atom] = point[0];
        flat[3 * atom + 1] = point[1];
        flat[3 * atom + 2] = point[2];
    }
    let r1 = 1.0;
    let r2 = 2.2;
    let mut diameter: f64 = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d2 = length2([
                points[j][0] - points[i][0],
                points[j][1] - points[i][1],
                points[j][2] - points[i][2],
            ]);
            diameter = diameter.max(d2.sqrt());
        }
    }
    assert!(
        diameter < r2,
        "the cluster diameter {diameter} sits inside {r2}"
    );
    let (h1, h2) = coordination_shell_counts(flat.view(), r1, r2);
    let mut expect1 = vec![0usize; n];
    let mut expect2 = vec![0usize; n];
    let r1sq = r1 * r1;
    let r2sq = r2 * r2;
    for i in 0..n {
        let mut n1 = 0usize;
        let mut n2 = 0usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d2 = length2([
                points[j][0] - points[i][0],
                points[j][1] - points[i][1],
                points[j][2] - points[i][2],
            ]);
            if d2 <= r1sq {
                n1 += 1;
            } else if d2 <= r2sq {
                n2 += 1;
            }
        }
        expect1[n1] += 1;
        expect2[n2] += 1;
    }
    assert_eq!(h1, expect1);
    assert_eq!(h2, expect2);
}

/// The previous descriptor loop: fractional wrap, then extra lattice shifts.
/// A zero shift is that wrap. Later shifts are images of it.
fn hand_rolled_images(
    positions: &[[f64; 3]],
    cell: &Cell,
    cutoff: f64,
) -> Vec<Vec<CutoffNeighbour>> {
    let n = positions.len();
    let widths = cell.widths();
    let mut bounds = [0_i32; 3];
    for axis in 0..3 {
        bounds[axis] = (cutoff / widths[axis] + 0.5).ceil() as i32;
    }
    let mut lists = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            let base = cell.displacement(positions[i], positions[j]);
            for na in -bounds[0]..=bounds[0] {
                for nb in -bounds[1]..=bounds[1] {
                    for nc in -bounds[2]..=bounds[2] {
                        if i == j && na == 0 && nb == 0 && nc == 0 {
                            continue;
                        }
                        let shift = cell.lattice_shift(na, nb, nc);
                        let displacement =
                            [base[0] + shift[0], base[1] + shift[1], base[2] + shift[2]];
                        let distance = length2(displacement).sqrt();
                        if distance <= 1e-12 || distance >= cutoff {
                            continue;
                        }
                        lists[i].push(CutoffNeighbour {
                            index: j,
                            displacement,
                            distance,
                        });
                    }
                }
            }
        }
    }
    lists
}

fn assert_descriptor_matches(rows: &[Vec<(usize, [f64; 3])>], expect: &[Vec<CutoffNeighbour>]) {
    assert_eq!(rows.len(), expect.len());
    for (centre, (found, wanted)) in rows.iter().zip(expect.iter()).enumerate() {
        assert_eq!(found.len(), wanted.len(), "centre {centre}");
        for right in wanted {
            let left = found
                .iter()
                .find(|neighbour| neighbour.0 == right.index)
                .unwrap_or_else(|| panic!("centre {centre} missing atom {}", right.index));
            for axis in 0..3 {
                assert!(
                    (left.1[axis] - right.displacement[axis]).abs() < 1e-9,
                    "centre {centre} atom {} axis {axis}: got {:?} want {:?}",
                    right.index,
                    left.1,
                    right.displacement
                );
            }
        }
    }
}

#[test]
fn descriptor_periodic_neighbours_use_the_shortest_vector() {
    let a = [10.0, 0.0, 0.0];
    let b = [5.0, 8.660254037844386, 0.0];
    let c = [0.0, 0.0, 10.0];
    let cell = Cell::from_vectors(a, b, c, [0.0; 3]).expect("hexagonal cell");
    let positions = [cell.cartesian([0.49, 0.49, 0.49]), [0.0; 3]];
    let cutoff = 9.85;
    let geometry = DescriptorGeometry::new(
        1.0,
        Some([a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]),
        [true; 3],
    )
    .expect("hexagonal descriptor cell");
    let coordinates = Array1::from_vec(vec![
        positions[0][0],
        positions[0][1],
        positions[0][2],
        positions[1][0],
        positions[1][1],
        positions[1][2],
    ]);
    let rows = descriptor_cutoff_neighbours(geometry, coordinates.view(), cutoff)
        .expect("descriptor neighbours");
    let vectors = [a, b, c];
    let shipped = cutoff_pairs(&positions, vectors, [true; 3], cutoff).expect("cutoff pairs");
    let expect = brute_shortest(&positions, &cell, [true; 3], cutoff);
    assert_lists_match(&shipped, &expect);
    assert_descriptor_matches(&rows, &expect);
    let named = rows[0]
        .iter()
        .find(|neighbour| neighbour.0 == 1)
        .expect("pair (0, 1)");
    let fractional = cell.displacement(positions[0], positions[1]);
    let old = hand_rolled_images(&positions, &cell, cutoff);
    let old_pair: Vec<_> = old[0].iter().filter(|image| image.index == 1).collect();
    assert!(
        old_pair.len() > 1,
        "the hand-rolled loop keeps more than one image of pair (0, 1)"
    );
    assert!(
        old_pair.iter().any(|image| {
            (image.displacement[0] - fractional[0]).abs() < 1e-8
                && (image.displacement[1] - fractional[1]).abs() < 1e-8
                && (image.displacement[2] - fractional[2]).abs() < 1e-8
        }),
        "pair (0, 1) hand-rolled loop still contains the fractional wrap {:?}",
        fractional
    );
    assert!(
        length2(fractional) > length2(named.1) + 1e-8,
        "pair (0, 1) keeps the shortest vector {:?}, not the fractional wrap {:?}",
        named.1,
        fractional
    );
}
