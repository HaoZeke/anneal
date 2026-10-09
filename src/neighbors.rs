//! An incremental neighbour table shared across the hop.
//!
//! Every kernel that reads structure recomputes it per call: surface
//! relocation its coordination counts, the graph key its distances, ring
//! profiles their adjacency. Each is O(n^2) per proposal, and a hop makes
//! several proposals from the same incumbent. The measured-productive moves
//! displace one to three atoms, so between incumbents the table changes in
//! O(k n), not O(n^2): the blocked-kernel economy of the eigensolver
//! libraries, applied to the structure every consumer shares.
//!
//! The table is exact, not approximate: after any sequence of updates it
//! equals the table built from scratch, and a test witnesses that on random
//! configurations and random moves.

use ndarray::ArrayView1;

/// Sorted adjacency lists under a fixed absolute cutoff.
#[derive(Debug, Clone)]
pub struct NeighborTable {
    cutoff2: f64,
    lists: Vec<Vec<usize>>,
}

impl NeighborTable {
    /// Builds the table. Prefers vesin cell lists when that backend is
    /// compiled in; otherwise the exact all-pairs construction.
    pub fn build(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Self {
        #[cfg(feature = "vesin-nl")]
        {
            if let Some(table) = Self::build_vesin(x, n, cutoff) {
                return table;
            }
        }
        Self::build_naive(x, n, cutoff)
    }

    /// Exact all-pairs construction.
    pub fn build_naive(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Self {
        let cutoff2 = cutoff * cutoff;
        let mut lists = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                if Self::dist2(x, i, j) < cutoff2 {
                    lists[i].push(j);
                    lists[j].push(i);
                }
            }
        }
        Self { cutoff2, lists }
    }

    fn dist2(x: ArrayView1<f64>, i: usize, j: usize) -> f64 {
        (0..3)
            .map(|k| {
                let d = x[3 * i + k] - x[3 * j + k];
                d * d
            })
            .sum()
    }

    /// Points in the table.
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }

    /// Neighbours of `i`, sorted ascending.
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.lists[i]
    }

    /// Coordination of `i`.
    pub fn degree(&self, i: usize) -> usize {
        self.lists[i].len()
    }

    /// Reconciles the table with `x_new` after the atoms in `moved` changed,
    /// in O(k n) for k moved atoms.
    ///
    /// Exactness rests on one fact: a pair's distance changed only if at
    /// least one of its ends moved, so edges between unmoved atoms need no
    /// inspection.
    pub fn update(&mut self, x_new: ArrayView1<f64>, moved: &[usize]) {
        let n = self.lists.len();
        let mut is_moved = vec![false; n];
        for &m in moved {
            if m < n {
                is_moved[m] = true;
            }
        }
        // Drop every edge with a moved end.
        for i in 0..n {
            if is_moved[i] {
                self.lists[i].clear();
            } else {
                self.lists[i].retain(|&j| !is_moved[j]);
            }
        }
        // Rebuild the moved atoms' edges against everyone.
        for &m in moved {
            if m >= n {
                continue;
            }
            for j in 0..n {
                if j == m || (is_moved[j] && j < m) {
                    continue;
                }
                if Self::dist2(x_new, m, j) < self.cutoff2 {
                    self.lists[m].push(j);
                    self.lists[j].push(m);
                }
            }
        }
        for l in self.lists.iter_mut() {
            l.sort_unstable();
        }
    }

    /// The atoms whose coordinates differ between two structures.
    pub fn moved_between(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Vec<usize> {
        let n = a.len().min(b.len()) / 3;
        (0..n)
            .filter(|&i| (0..3).any(|k| (a[3 * i + k] - b[3 * i + k]).abs() > 1e-12))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// After any sequence of moves, the incremental table must equal the one
    /// built from scratch. This is the whole contract.
    #[test]
    fn incremental_equals_rebuild() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 40;
        let mut x = Array1::from_shape_fn(3 * n, |_| rng.random::<f64>() * 4.0);
        let mut table = NeighborTable::build(x.view(), n, 1.4);
        for round in 0..30 {
            let k = 1 + rng.random_range(0..3);
            let mut moved = Vec::new();
            for _ in 0..k {
                let m = rng.random_range(0..n);
                moved.push(m);
                for c in 0..3 {
                    x[3 * m + c] += (rng.random::<f64>() - 0.5) * 2.0;
                }
            }
            moved.sort_unstable();
            moved.dedup();
            table.update(x.view(), &moved);
            let fresh = NeighborTable::build(x.view(), n, 1.4);
            for i in 0..n {
                assert_eq!(
                    table.neighbors(i),
                    fresh.neighbors(i),
                    "round {round}, atom {i}: incremental diverged from rebuild"
                );
            }
        }
    }

    /// The moved-set detector has to find exactly the atoms that differ.
    #[test]
    fn moved_between_finds_the_difference() {
        let mut rng = StdRng::seed_from_u64(9);
        let n = 20;
        let a = Array1::from_shape_fn(3 * n, |_| rng.random::<f64>());
        let mut b = a.clone();
        b[3 * 7] += 0.5;
        b[3 * 13 + 2] -= 0.1;
        assert_eq!(
            NeighborTable::moved_between(a.view(), b.view()),
            vec![7, 13]
        );
    }
}

/// vesin-backed construction: the cell lists of the metatensor ecosystem's
/// neighbour library, compiled from its own sources rather than reimplemented.
#[cfg(feature = "vesin-nl")]
mod vesin_ffi {
    use super::NeighborTable;
    use ndarray::ArrayView1;

    #[repr(C)]
    struct VesinOptions {
        cutoff: f64,
        full: bool,
        sorted: bool,
        algorithm: i32,
        skin: f64,
        n_threads: i32,
        return_shifts: bool,
        return_distances: bool,
        return_vectors: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct VesinDevice {
        kind: i32,
        device_id: i32,
    }

    #[repr(C)]
    struct VesinNeighborList {
        length: usize,
        device: VesinDevice,
        pairs: *mut [usize; 2],
        shifts: *mut [i32; 3],
        distances: *mut f64,
        vectors: *mut [f64; 3],
        opaque: *mut core::ffi::c_void,
    }

    unsafe extern "C" {
        fn vesin_neighbors(
            points: *const [f64; 3],
            n_points: usize,
            bounding_box: *const [f64; 3],
            periodic: *const bool,
            device: VesinDevice,
            options: VesinOptions,
            neighbors: *mut VesinNeighborList,
            error_message: *mut *const core::ffi::c_char,
        ) -> i32;
        fn vesin_free(neighbors: *mut VesinNeighborList);
    }

    impl NeighborTable {
        /// Builds the table through vesin's cell lists.
        pub fn build_vesin(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Option<Self> {
            let points: Vec<[f64; 3]> = (0..n)
                .map(|i| [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
                .collect();
            let bbox = [[0.0; 3]; 3];
            let periodic = [false; 3];
            let options = VesinOptions {
                cutoff,
                full: true,
                sorted: false,
                algorithm: 0,
                skin: 0.0,
                n_threads: 1,
                return_shifts: false,
                return_distances: false,
                return_vectors: false,
            };
            let mut list = VesinNeighborList {
                length: 0,
                device: VesinDevice {
                    kind: 0,
                    device_id: 0,
                },
                pairs: core::ptr::null_mut(),
                shifts: core::ptr::null_mut(),
                distances: core::ptr::null_mut(),
                vectors: core::ptr::null_mut(),
                opaque: core::ptr::null_mut(),
            };
            let mut err: *const core::ffi::c_char = core::ptr::null();
            let status = unsafe {
                vesin_neighbors(
                    points.as_ptr(),
                    n,
                    bbox.as_ptr(),
                    periodic.as_ptr(),
                    VesinDevice {
                        kind: 1,
                        device_id: 0,
                    },
                    options,
                    &mut list,
                    &mut err,
                )
            };
            if status != 0 {
                return None;
            }
            let mut lists = vec![Vec::new(); n];
            let pairs = unsafe { core::slice::from_raw_parts(list.pairs, list.length) };
            for p in pairs {
                let (a, b) = (p[0], p[1]);
                if a < n && b < n && a != b {
                    lists[a].push(b);
                }
            }
            unsafe { vesin_free(&mut list) };
            for l in lists.iter_mut() {
                l.sort_unstable();
            }
            Some(Self {
                cutoff2: cutoff * cutoff,
                lists,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::NeighborTable;
        use ndarray::Array1;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        /// vesin's lists must equal the naive build exactly, or the backend
        /// is not a backend.
        #[test]
        fn vesin_equals_naive() {
            let mut rng = StdRng::seed_from_u64(17);
            for _ in 0..5 {
                let n = 60;
                let x = Array1::from_shape_fn(3 * n, |_| rng.random::<f64>() * 5.0);
                let naive = NeighborTable::build_naive(x.view(), n, 1.4);
                let fast = NeighborTable::build_vesin(x.view(), n, 1.4).expect("vesin refused");
                for i in 0..n {
                    assert_eq!(naive.neighbors(i), fast.neighbors(i), "atom {i}");
                }
            }
        }
    }

    /// Cutoff rows for an open or partly open cell. An axis marked open is
    /// not wrapped. Each ordered pair keeps its shortest image, including a
    /// periodic self-image. `inclusive` keeps a pair whose distance equals
    /// `cutoff`.
    pub(super) fn cutoff_rows(
        positions: &[[f64; 3]],
        vectors: Option<[[f64; 3]; 3]>,
        periodic: [bool; 3],
        cutoff: f64,
        inclusive: bool,
    ) -> Option<Vec<Vec<super::CutoffNeighbour>>> {
        let n = positions.len();
        let bbox = vectors.unwrap_or([[0.0; 3]; 3]);
        let query = if inclusive { cutoff.next_up() } else { cutoff };
        let options = VesinOptions {
            cutoff: query,
            full: true,
            sorted: false,
            algorithm: 0,
            skin: 0.0,
            n_threads: 1,
            return_shifts: true,
            return_distances: true,
            return_vectors: true,
        };
        let mut list = VesinNeighborList {
            length: 0,
            device: VesinDevice {
                kind: 0,
                device_id: 0,
            },
            pairs: core::ptr::null_mut(),
            shifts: core::ptr::null_mut(),
            distances: core::ptr::null_mut(),
            vectors: core::ptr::null_mut(),
            opaque: core::ptr::null_mut(),
        };
        let mut err: *const core::ffi::c_char = core::ptr::null();
        let status = unsafe {
            vesin_neighbors(
                positions.as_ptr(),
                n,
                bbox.as_ptr(),
                periodic.as_ptr(),
                VesinDevice {
                    kind: 1,
                    device_id: 0,
                },
                options,
                &mut list,
                &mut err,
            )
        };
        if status != 0 {
            return None;
        }
        let pairs = unsafe { core::slice::from_raw_parts(list.pairs, list.length) };
        let returned = unsafe { core::slice::from_raw_parts(list.vectors, list.length) };
        let distances = unsafe { core::slice::from_raw_parts(list.distances, list.length) };
        let limit = cutoff * cutoff;
        let mut best = vec![vec![None; n]; n];
        for (pair, (vector, distance)) in pairs.iter().zip(returned.iter().zip(distances.iter())) {
            let (i, j) = (pair[0], pair[1]);
            if i >= n || j >= n {
                continue;
            }
            let d2 = distance * distance;
            let keep = if inclusive { d2 <= limit } else { d2 < limit };
            if !keep || d2 <= 1e-24 {
                continue;
            }
            super::consider(&mut best[i][j], d2, *vector);
        }
        unsafe { vesin_free(&mut list) };
        let mut rows = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                let Some((length2, displacement)) = best[i][j] else {
                    continue;
                };
                rows[i].push(super::CutoffNeighbour {
                    index: j,
                    displacement,
                    distance: length2.sqrt(),
                });
            }
            rows[i].sort_by(|left, right| {
                left.index
                    .cmp(&right.index)
                    .then_with(|| left.displacement[0].total_cmp(&right.displacement[0]))
            });
        }
        Some(rows)
    }
}

/// One neighbour inside a cutoff, with the displacement from the centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutoffNeighbour {
    /// Index of the neighbour atom.
    pub index: usize,
    /// Cartesian displacement from the centre to the chosen image.
    pub displacement: [f64; 3],
    /// Length of `displacement`.
    pub distance: f64,
}

/// Periodic cutoff list.
///
/// The pairs are [`linkcell::pairs_within`]: every lattice image inside
/// the cutoff, including a periodic self-image. Each displacement is that
/// image, `q - p + lattice_shift`. A half-box image at `+L/2` is also
/// present as `-L/2`.
pub fn periodic_cutoff_pairs(
    positions: &[[f64; 3]],
    cell: &linkcell::Cell,
    cutoff: f64,
) -> Result<Vec<Vec<CutoffNeighbour>>, linkcell::Error> {
    let n = positions.len();
    let mut lists = vec![Vec::new(); n];
    if n == 0 || !(cutoff.is_finite() && cutoff > 0.0) {
        return Ok(lists);
    }
    let cutoff2 = cutoff * cutoff;
    let rows = linkcell::pairs_within(positions, cell, cutoff, None, None, false)?;
    for row in rows {
        if row.i >= n || row.j >= n {
            continue;
        }
        let shift = cell.lattice_shift(row.shift[0], row.shift[1], row.shift[2]);
        let p = positions[row.i];
        let q = positions[row.j];
        let displacement = [
            q[0] + shift[0] - p[0],
            q[1] + shift[1] - p[1],
            q[2] + shift[2] - p[2],
        ];
        let length2 = dot3(displacement);
        if !(length2 < cutoff2) || length2 <= 1e-24 {
            continue;
        }
        lists[row.i].push(CutoffNeighbour {
            index: row.j,
            displacement,
            distance: length2.sqrt(),
        });
    }
    for list in &mut lists {
        list.sort_by(|left, right| {
            left.index
                .cmp(&right.index)
                .then_with(|| left.displacement[0].total_cmp(&right.displacement[0]))
                .then_with(|| left.displacement[1].total_cmp(&right.displacement[1]))
                .then_with(|| left.displacement[2].total_cmp(&right.displacement[2]))
        });
    }
    Ok(lists)
}

/// Cutoff pairs on a free cluster. No coordinate is wrapped.
///
/// When the vesin backend is compiled, the rows come from vesin. Otherwise
/// the same Cartesian pairs are counted directly. `inclusive` keeps a
/// distance that equals `cutoff`.
pub fn open_cutoff_pairs(
    positions: &[[f64; 3]],
    cutoff: f64,
    inclusive: bool,
) -> Vec<Vec<CutoffNeighbour>> {
    #[cfg(feature = "vesin-nl")]
    if let Some(rows) = vesin_ffi::cutoff_rows(positions, None, [false; 3], cutoff, inclusive) {
        return rows;
    }
    cartesian_pairs(positions, cutoff, inclusive)
}

/// Cutoff pairs for a cell that may leave an axis open.
///
/// Every periodic axis uses [`periodic_cutoff_pairs`]. An open axis uses
/// [`open_cutoff_pairs`] and is not wrapped. A cell with both kinds of axis
/// searches lattice images on the periodic axes only.
pub fn cutoff_pairs(
    positions: &[[f64; 3]],
    vectors: [[f64; 3]; 3],
    periodic: [bool; 3],
    cutoff: f64,
) -> Result<Vec<Vec<CutoffNeighbour>>, linkcell::Error> {
    if periodic.iter().all(|axis| *axis) {
        let cell = linkcell::Cell::from_vectors(vectors[0], vectors[1], vectors[2], [0.0; 3])?;
        return periodic_cutoff_pairs(positions, &cell, cutoff);
    }
    if periodic.iter().all(|axis| !*axis) {
        return Ok(open_cutoff_pairs(positions, cutoff, false));
    }
    #[cfg(feature = "vesin-nl")]
    if let Some(rows) = vesin_ffi::cutoff_rows(positions, Some(vectors), periodic, cutoff, false) {
        return Ok(rows);
    }
    mixed_cutoff_pairs(positions, vectors, periodic, cutoff)
}

fn cartesian_pairs(
    positions: &[[f64; 3]],
    cutoff: f64,
    inclusive: bool,
) -> Vec<Vec<CutoffNeighbour>> {
    let n = positions.len();
    let mut lists = vec![Vec::new(); n];
    if n == 0 || !(cutoff.is_finite() && cutoff > 0.0) {
        return lists;
    }
    let limit = cutoff * cutoff;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let displacement = [
                positions[j][0] - positions[i][0],
                positions[j][1] - positions[i][1],
                positions[j][2] - positions[i][2],
            ];
            let length2 = dot3(displacement);
            let keep = if inclusive {
                length2 <= limit
            } else {
                length2 < limit
            };
            if keep && length2 > 1e-24 {
                lists[i].push(CutoffNeighbour {
                    index: j,
                    displacement,
                    distance: length2.sqrt(),
                });
            }
        }
    }
    lists
}

fn mixed_cutoff_pairs(
    positions: &[[f64; 3]],
    vectors: [[f64; 3]; 3],
    periodic: [bool; 3],
    cutoff: f64,
) -> Result<Vec<Vec<CutoffNeighbour>>, linkcell::Error> {
    let n = positions.len();
    let mut lists = vec![Vec::new(); n];
    if n == 0 || !(cutoff.is_finite() && cutoff > 0.0) {
        return Ok(lists);
    }
    let cell = linkcell::Cell::from_vectors(vectors[0], vectors[1], vectors[2], [0.0; 3])?;
    let widths = cell.widths();
    let mut reach = [0_i32; 3];
    for axis in 0..3 {
        if !periodic[axis] {
            continue;
        }
        let width = widths[axis].abs().max(1e-12);
        let images = (cutoff / width).ceil() as i32 + 1;
        reach[axis] = images.clamp(1, 8);
    }
    let cutoff2 = cutoff * cutoff;
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
                        let length2 = dot3(displacement);
                        if length2 < cutoff2 && length2 > 1e-24 {
                            consider(&mut chosen, length2, displacement);
                        }
                    }
                }
            }
            if let Some((_, displacement)) = chosen {
                lists[i].push(CutoffNeighbour {
                    index: j,
                    displacement,
                    distance: dot3(displacement).sqrt(),
                });
            }
        }
        lists[i].sort_by(|left, right| left.index.cmp(&right.index));
    }
    Ok(lists)
}

fn consider(slot: &mut Option<(f64, [f64; 3])>, length2: f64, displacement: [f64; 3]) {
    let replace = match slot {
        None => true,
        Some((old2, old)) => {
            if length2 < *old2 - 1e-12 {
                true
            } else if (*old2 - length2).abs() <= 1e-9 * old2.max(1.0) {
                prefers_negative(displacement, *old)
            } else {
                false
            }
        }
    };
    if replace {
        *slot = Some((length2, displacement));
    }
}

/// On a length tie, keep the vector whose first differing component is
/// smaller. An orthorhombic half-box pair then keeps `-L/2` rather than
/// `+L/2`.
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

fn dot3(value: [f64; 3]) -> f64 {
    value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
}
