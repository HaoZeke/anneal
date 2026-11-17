//! Pair candidates from linkcell and minimum-image vectors from minimage.
//!
//! The C entry points are what the classical pots call. A failure is a
//! negative code plus a thread-local message. There is no second search
//! behind these functions.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

use linkcell::{pairs_within, wrap_many, Cell};

/// Bins of at least one cutoff that a periodic axis needs before a
/// linked-cell walk pays. Each bin is checked against itself and its 26
/// neighbours. With one or two bins along an axis, the neighbours on
/// either side are the same bin, and the walk visits a pair more than
/// once. Allen and Tildesley, Computer Simulation of Liquids, 2nd ed.
/// (2017), section 5.3.2, stop their cell-list codes below three.
const MIN_BINS_PER_PERIODIC_AXIS: f64 = 3.0;

/// Width added to an open axis beyond its extent plus two cutoffs, half
/// on each side. An atom's image across the padded axis is then more
/// than one cutoff away, for any positive margin.
const OPEN_AXIS_MARGIN: f64 = 1.0;

/// Width given to an open axis in the axis wrap. A component folds only
/// once its magnitude reaches half this width.
const OPEN_AXIS_WRAP_WIDTH: f64 = 1.0e16;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

fn fail(msg: &str) -> i32 {
    let text = msg.replace('\0', " ");
    let owned = CString::new(text).unwrap_or_else(|_| CString::new("fastnlist error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(owned));
    -1
}

/// Thread-local message for the most recent negative return.
///
/// # Safety
///
/// The pointer is valid until the next fastnlist call on this thread.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(msg) => msg.as_ptr(),
        None => std::ptr::null(),
    })
}

fn triples<'a>(ptr: *const f64, n: usize) -> Result<&'a [[f64; 3]], i32> {
    if n == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(fail("null coordinate buffer"));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const [f64; 3], n) })
}

fn cell_rows(cell9: *const f64) -> Result<[[f64; 3]; 3], i32> {
    if cell9.is_null() {
        return Err(fail("null cell"));
    }
    let s = unsafe { std::slice::from_raw_parts(cell9, 9) };
    Ok([
        [s[0], s[1], s[2]],
        [s[3], s[4], s[5]],
        [s[6], s[7], s[8]],
    ])
}

fn periodic3(periodic: *const i32) -> Result<[bool; 3], i32> {
    if periodic.is_null() {
        return Err(fail("null periodic flags"));
    }
    let s = unsafe { std::slice::from_raw_parts(periodic, 3) };
    Ok([s[0] != 0, s[1] != 0, s[2] != 0])
}

/// False for zero, negatives and NaN.
fn positive(x: f64) -> bool {
    x > 0.0
}

fn orthorhombic(cell: &[[f64; 3]; 3]) -> bool {
    cell[0][1] == 0.0
        && cell[0][2] == 0.0
        && cell[1][0] == 0.0
        && cell[1][2] == 0.0
        && cell[2][0] == 0.0
        && cell[2][1] == 0.0
}

/// Candidate pairs. See `fastnlist.h`.
///
/// # Safety
///
/// `xyz` is `n * 3` doubles. `cell9` is 9 doubles. `periodic` is 3 ints.
/// `out_count` is writable. On a fill, `out_ij` holds `2 * cap` ints.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_pairs(
    xyz: *const f64,
    n: usize,
    cell9: *const f64,
    periodic: *const i32,
    list_cutoff: f64,
    out_ij: *mut i32,
    cap: usize,
    out_count: *mut usize,
) -> i32 {
    if out_count.is_null() {
        return fail("null out_count");
    }
    let points = match triples(xyz, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let periodic = match periodic3(periodic) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let found = match collect_pairs(points, &cell, periodic, list_cutoff) {
        Ok(found) => found,
        Err(code) => return code,
    };
    let Some(count) = found.count() else {
        return fail("pair count does not fit in size_t");
    };
    unsafe { *out_count = count };
    if count > cap {
        clear_error();
        return 1;
    }
    if count > 0 {
        if out_ij.is_null() {
            return fail("null out_ij");
        }
        found.write(unsafe { std::slice::from_raw_parts_mut(out_ij as *mut [i32; 2], count) });
    }
    clear_error();
    0
}

/// Pair candidates from `collect_pairs`.
enum Candidates {
    /// Every unordered pair of this many atoms, `(0, 1), (0, 2), ...,
    /// (n - 2, n - 1)`. Held as the count so a short buffer is answered
    /// without building the list.
    Every(usize),
    /// Unique pairs with `i < j`, sorted.
    Listed(Vec<(i32, i32)>),
}

impl Candidates {
    /// Number of pairs, or `None` when it overflows `usize`.
    fn count(&self) -> Option<usize> {
        match self {
            Candidates::Every(n) => n.checked_mul(n.saturating_sub(1)).map(|m| m / 2),
            Candidates::Listed(pairs) => Some(pairs.len()),
        }
    }

    /// Writes `[i, j]` for each pair into `out`, which holds `count()`
    /// slots.
    fn write(&self, out: &mut [[i32; 2]]) {
        match self {
            Candidates::Every(n) => {
                let mut slots = out.iter_mut();
                for i in 0..*n {
                    for j in (i + 1)..*n {
                        if let Some(slot) = slots.next() {
                            *slot = [i as i32, j as i32];
                        }
                    }
                }
            }
            Candidates::Listed(pairs) => {
                for (slot, (i, j)) in out.iter_mut().zip(pairs) {
                    *slot = [*i, *j];
                }
            }
        }
    }
}

/// True when a linked-cell walk visits fewer images than one pass over
/// every unordered pair. A periodic axis narrower than three cutoffs has
/// one or two bins and a fat stencil. An open axis whose points all lie
/// inside the cutoff has nothing to reject.
fn linkcell_helps(
    widths: &[f64; 3],
    periodic: &[bool; 3],
    mins: &[f64; 3],
    maxs: &[f64; 3],
    list_cutoff: f64,
) -> bool {
    for k in 0..3 {
        if periodic[k] {
            if widths[k] < MIN_BINS_PER_PERIODIC_AXIS * list_cutoff {
                return false;
            }
        } else if maxs[k] - mins[k] <= list_cutoff {
            return false;
        }
    }
    true
}

fn collect_pairs(
    xyz: &[[f64; 3]],
    cell: &[[f64; 3]; 3],
    periodic: [bool; 3],
    list_cutoff: f64,
) -> Result<Candidates, i32> {
    if xyz.len() < 2 || !positive(list_cutoff) || !list_cutoff.is_finite() {
        return Ok(Candidates::Listed(Vec::new()));
    }
    if !orthorhombic(cell) {
        return Err(fail("pair list requires an orthorhombic cell"));
    }
    if xyz.len() > i32::MAX as usize {
        return Err(fail("atom count does not fit in i32"));
    }
    let mut widths = [cell[0][0], cell[1][1], cell[2][2]];
    let mut shift = [0.0; 3];
    let mut any_open = false;
    let mut mins = [f64::INFINITY; 3];
    let mut maxs = [f64::NEG_INFINITY; 3];
    for p in xyz {
        for k in 0..3 {
            mins[k] = mins[k].min(p[k]);
            maxs[k] = maxs[k].max(p[k]);
        }
    }
    for k in 0..3 {
        if periodic[k] {
            if !positive(widths[k]) {
                return Err(fail("periodic axis has a non-positive width"));
            }
        } else {
            any_open = true;
            let extent = maxs[k] - mins[k];
            widths[k] = extent + 2.0 * list_cutoff + OPEN_AXIS_MARGIN;
            if !positive(widths[k]) || !widths[k].is_finite() {
                return Err(fail("open axis padding is not finite"));
            }
            shift[k] = list_cutoff + 0.5 * OPEN_AXIS_MARGIN - mins[k];
        }
    }
    // One bin, or a cutoff that reaches the neighbouring image, makes the
    // linked-cell stencil revisit every pair. Emitting i < j and letting
    // the caller fold with minimage is the shorter walk, at any atom
    // count. The 10k-atom bridge cell is narrower than three cutoffs;
    // asking linkcell for that stencil allocates tens of gigabytes, and
    // even its n(n-1)/2 pairs are 400 MB, so they go straight into the
    // caller's buffer. Wider cells still go through linkcell.
    if !linkcell_helps(&widths, &periodic, &mins, &maxs, list_cutoff) {
        return Ok(Candidates::Every(xyz.len()));
    }
    let owned;
    let pts: &[[f64; 3]] = if any_open {
        let mut moved = xyz.to_vec();
        for p in &mut moved {
            for k in 0..3 {
                if !periodic[k] {
                    p[k] += shift[k];
                }
            }
        }
        owned = moved;
        &owned
    } else {
        xyz
    };
    let sim = match Cell::ortho(widths[0], widths[1], widths[2]) {
        Ok(c) => c,
        Err(err) => return Err(fail(&err.to_string())),
    };
    let rows = match pairs_within(pts, &sim, list_cutoff, None, None, true) {
        Ok(rows) => rows,
        Err(err) => return Err(fail(&err.to_string())),
    };
    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        if row.i == row.j {
            continue;
        }
        let (i, j) = if row.i < row.j {
            (row.i, row.j)
        } else {
            (row.j, row.i)
        };
        pairs.push((i as i32, j as i32));
    }
    pairs.sort_unstable();
    pairs.dedup();
    Ok(Candidates::Listed(pairs))
}

fn wrap_into(cell: &Cell, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) -> Result<(), i32> {
    wrap_many(cell, diffs, out).map_err(|err| fail(&err.to_string()))
}

/// Full-cell minimum-image wrap. See `fastnlist.h`.
///
/// # Safety
///
/// `cell9` is 9 doubles. `diffs` and `out` are `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_wrap(
    cell9: *const f64,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let diffs = match triples(diffs, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    if n > 0 && out.is_null() {
        return fail("null wrap output");
    }
    let sim = match Cell::from_vesin(cell) {
        Ok(c) => c,
        Err(err) => return fail(&err.to_string()),
    };
    let out = if n == 0 {
        &mut [][..]
    } else {
        unsafe { std::slice::from_raw_parts_mut(out as *mut [f64; 3], n) }
    };
    match wrap_into(&sim, diffs, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(code) => code,
    }
}

/// Orthorhombic wrap that skips open axes. See `fastnlist.h`.
///
/// # Safety
///
/// `cell9` is 9 doubles. `periodic` is 3 ints. `diffs` and `out` are
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_wrap_axes(
    cell9: *const f64,
    periodic: *const i32,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let periodic = match periodic3(periodic) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let diffs = match triples(diffs, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    if n > 0 && out.is_null() {
        return fail("null wrap output");
    }
    if !orthorhombic(&cell) {
        return fail("axis wrap requires an orthorhombic cell");
    }
    // An open axis gets a width no difference in this library can cross,
    // so the orthorhombic wrap leaves that component alone.
    let mut widths = [cell[0][0], cell[1][1], cell[2][2]];
    for k in 0..3 {
        if periodic[k] {
            if !positive(widths[k]) {
                return fail("periodic axis has a non-positive width");
            }
        } else {
            widths[k] = OPEN_AXIS_WRAP_WIDTH;
        }
    }
    let sim = match Cell::ortho(widths[0], widths[1], widths[2]) {
        Ok(c) => c,
        Err(err) => return fail(&err.to_string()),
    };
    let out = if n == 0 {
        &mut [][..]
    } else {
        unsafe { std::slice::from_raw_parts_mut(out as *mut [f64; 3], n) }
    };
    match wrap_into(&sim, diffs, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};

    /// Counts the bytes each thread asks for, so a test can bound what one
    /// call allocates while other tests run on their own threads.
    struct Counting;

    thread_local! {
        static REQUESTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = REQUESTED.try_with(|b| b.set(b.get().saturating_add(layout.size())));
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static COUNTING: Counting = Counting;

    fn requested_during<R>(f: impl FnOnce() -> R) -> (R, usize) {
        let before = REQUESTED.with(|b| b.get());
        let out = f();
        (out, REQUESTED.with(|b| b.get()) - before)
    }

    fn pairs_of(
        xyz: &[[f64; 3]],
        cell: &[[f64; 3]; 3],
        periodic: [bool; 3],
        list_cutoff: f64,
    ) -> Vec<(i32, i32)> {
        let found = collect_pairs(xyz, cell, periodic, list_cutoff).unwrap();
        let mut slots = vec![[0; 2]; found.count().unwrap()];
        found.write(&mut slots);
        slots.iter().map(|&[i, j]| (i, j)).collect()
    }

    /// The client bridge stress cell: a 10 A box under the default
    /// Lennard-Jones cutoff of 15 A, one bin per axis.
    fn bridge_cell(n: usize) -> (Vec<[f64; 3]>, [f64; 9]) {
        let xyz = (0..n)
            .map(|i| {
                let t = i as f64;
                [
                    (t * 0.37).rem_euclid(10.0),
                    (t * 0.53).rem_euclid(10.0),
                    (t * 0.71).rem_euclid(10.0),
                ]
            })
            .collect();
        (xyz, [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0])
    }

    const BRIDGE_CUTOFF: f64 = 15.0;
    const ALL_PERIODIC: [i32; 3] = [1, 1, 1];

    fn round_away(t: f64) -> f64 {
        (t + 0.5_f64.copysign(t)) as i64 as f64
    }

    fn scalar_pairs(xyz: &[[f64; 3]], box_w: [f64; 3], periodic: [bool; 3], cutoff: f64) -> Vec<(i32, i32)> {
        let cut2 = cutoff * cutoff;
        let mut out = Vec::new();
        for i in 0..xyz.len() {
            for j in (i + 1)..xyz.len() {
                let mut d = [
                    xyz[i][0] - xyz[j][0],
                    xyz[i][1] - xyz[j][1],
                    xyz[i][2] - xyz[j][2],
                ];
                for k in 0..3 {
                    if periodic[k] {
                        let inv = 1.0 / box_w[k];
                        d[k] -= box_w[k] * round_away(d[k] * inv);
                    }
                }
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= cut2 {
                    out.push((i as i32, j as i32));
                }
            }
        }
        out
    }

    #[test]
    fn periodic_slab_matches_the_scalar_fold() {
        let mut xyz = Vec::new();
        for i in 0..12 {
            let t = i as f64;
            xyz.push([
                (t * 1.7).rem_euclid(18.0),
                (t * 2.3).rem_euclid(18.0),
                3.0 + (t * 0.4).sin(),
            ]);
        }
        // One pair sits across the x boundary.
        xyz.push([0.2, 9.0, 4.0]);
        xyz.push([17.7, 9.1, 4.2]);
        let cell = [[18.0, 0.0, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 30.0]];
        let periodic = [true, true, true];
        let cutoff = 6.0;
        let got = pairs_of(&xyz, &cell, periodic, cutoff + 1.0);
        let expect = scalar_pairs(&xyz, [18.0, 18.0, 30.0], periodic, cutoff + 1.0);
        // The box is only a couple of cutoffs wide, so the candidate
        // buffer is every unordered pair. The scalar set is the in-cutoff
        // subset the caller keeps.
        assert!(expect.iter().all(|pair| got.contains(pair)));
        assert_eq!(got.len(), xyz.len() * (xyz.len() - 1) / 2);
    }

    #[test]
    fn narrow_cell_above_the_old_cap_is_every_pair() {
        // Just past the old 4096 guard. A 10 A box with cutoff 15 A is the
        // bridge stress cell; linkcell must not see it.
        let n = 4100;
        let mut xyz = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            xyz.push([
                (t * 0.37).rem_euclid(10.0),
                (t * 0.53).rem_euclid(10.0),
                (t * 0.71).rem_euclid(10.0),
            ]);
        }
        let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let got = pairs_of(&xyz, &cell, [true, true, true], 15.0);
        assert_eq!(got.len(), n * (n - 1) / 2);
        assert_eq!(got.first().copied(), Some((0, 1)));
        assert_eq!(got.last().copied(), Some((n as i32 - 2, n as i32 - 1)));
    }

    #[test]
    fn wide_cell_returns_only_in_cutoff_pairs() {
        let mut xyz = Vec::new();
        for i in 0..20 {
            let t = i as f64;
            xyz.push([t * 1.6, (t * 0.7).sin() * 2.0 + 10.0, 5.0]);
        }
        let cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];
        let periodic = [true, true, true];
        let cutoff = 6.0;
        let got = pairs_of(&xyz, &cell, periodic, cutoff);
        let expect = scalar_pairs(&xyz, [40.0, 40.0, 40.0], periodic, cutoff);
        assert_eq!(got, expect);
        assert!(got.len() < xyz.len() * (xyz.len() - 1) / 2);
    }

    #[test]
    fn open_cluster_matches_raw_distances() {
        let xyz = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 8.0],
        ];
        let cell = [[1.0e6, 0.0, 0.0], [0.0, 1.0e6, 0.0], [0.0, 0.0, 1.0e6]];
        let got = pairs_of(&xyz, &cell, [false, false, false], 3.5);
        let expect = scalar_pairs(&xyz, [1.0e6, 1.0e6, 1.0e6], [false, false, false], 3.5);
        assert_eq!(expect, vec![(0, 1), (0, 2), (1, 2)]);
        assert!(expect.iter().all(|pair| got.contains(pair)));
    }

    #[test]
    fn wrap_matches_round_away_off_the_tie() {
        let cell = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]];
        let diffs = [[9.2, -0.4, 3.0], [-6.0, 7.0, -8.0], [0.0, 0.0, 0.0]];
        let mut out = [[0.0; 3]; 3];
        let sim = Cell::from_vesin(cell).unwrap();
        wrap_many(&sim, &diffs, &mut out).unwrap();
        for (d, w) in diffs.iter().zip(out.iter()) {
            let widths = [10.0, 12.0, 14.0];
            for k in 0..3 {
                let folded = d[k] - widths[k] * round_away(d[k] / widths[k]);
                assert!((w[k] - folded).abs() < 1e-12, "{} vs {}", w[k], folded);
            }
        }
    }

    fn fast_pairs(
        xyz: &[[f64; 3]],
        cell: &[f64; 9],
        list_cutoff: f64,
        out_ij: &mut [[i32; 2]],
        count: &mut usize,
    ) -> i32 {
        let out = if out_ij.is_empty() {
            std::ptr::null_mut()
        } else {
            out_ij.as_mut_ptr() as *mut i32
        };
        unsafe {
            rgpot_fast_pairs(
                xyz.as_ptr() as *const f64,
                xyz.len(),
                cell.as_ptr(),
                ALL_PERIODIC.as_ptr(),
                list_cutoff,
                out,
                out_ij.len(),
                count,
            )
        }
    }

    #[test]
    fn short_buffer_on_the_bridge_cell_is_answered_without_the_list() {
        let n = 10_000;
        let (xyz, cell) = bridge_cell(n);
        let mut count = 0;
        let (rc, bytes) =
            requested_during(|| fast_pairs(&xyz, &cell, BRIDGE_CUTOFF, &mut [], &mut count));
        assert_eq!(rc, 1);
        assert_eq!(count, n * (n - 1) / 2);
        assert!(unsafe { rgpot_fast_last_error() }.is_null());
        // A built list is about 400 MB here. The probe allocates less than
        // one copy of the coordinates.
        let coords = std::mem::size_of_val(&xyz[..]);
        assert!(bytes < coords, "{bytes} bytes");
    }

    #[test]
    fn fill_on_a_narrow_cell_writes_only_the_callers_buffer() {
        let n = 4100;
        let (xyz, cell) = bridge_cell(n);
        let cap = n * (n - 1) / 2;
        let mut ij = vec![[-1; 2]; cap];
        let mut count = 0;
        let (rc, bytes) =
            requested_during(|| fast_pairs(&xyz, &cell, BRIDGE_CUTOFF, &mut ij, &mut count));
        assert_eq!(rc, 0);
        assert_eq!(count, cap);
        let coords = std::mem::size_of_val(&xyz[..]);
        assert!(bytes < coords, "{bytes} bytes");
        let expect = (0..n as i32).flat_map(|i| ((i + 1)..n as i32).map(move |j| (i, j)));
        assert!(ij.iter().map(|&[i, j]| (i, j)).eq(expect));
    }

    #[test]
    fn every_pair_count_matches_the_built_list() {
        // A narrow cell takes the every-pair path. The probe reports the
        // count without a list; the fill and collect_pairs both build one.
        // Those three counts are the same number.
        let n = 64;
        let (xyz, cell) = bridge_cell(n);
        let rows = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let found = collect_pairs(&xyz, &rows, [true, true, true], BRIDGE_CUTOFF).unwrap();
        assert!(matches!(found, Candidates::Every(_)));
        let counted = found.count().unwrap();
        let mut slots = vec![[-1; 2]; counted];
        found.write(&mut slots);
        let built = slots.iter().take_while(|pair| **pair != [-1, -1]).count();
        assert_eq!(counted, built);

        let mut probed = 0;
        assert_eq!(fast_pairs(&xyz, &cell, BRIDGE_CUTOFF, &mut [], &mut probed), 1);
        let mut ij = vec![[-1; 2]; probed];
        let mut filled = 0;
        assert_eq!(fast_pairs(&xyz, &cell, BRIDGE_CUTOFF, &mut ij, &mut filled), 0);
        let written = ij.iter().take_while(|pair| **pair != [-1, -1]).count();
        assert_eq!(filled, written);
        assert_eq!(probed, written);
        assert_eq!(probed, built);
    }

    #[test]
    fn short_buffer_clears_an_earlier_error() {
        let (xyz, cell) = bridge_cell(3);
        let skewed = [10.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let mut count = 0;
        assert!(fast_pairs(&xyz, &skewed, BRIDGE_CUTOFF, &mut [], &mut count) < 0);
        assert!(!unsafe { rgpot_fast_last_error() }.is_null());
        let rc = fast_pairs(&xyz, &cell, BRIDGE_CUTOFF, &mut [], &mut count);
        assert_eq!(rc, 1);
        assert_eq!(count, 3);
        assert!(unsafe { rgpot_fast_last_error() }.is_null());
    }

    #[test]
    fn linkcell_probe_then_fill_matches_the_builder() {
        // 5 x 5 x 5 sites 5 A apart in a 25 A box. A 7 A list cutoff fits
        // three bins per axis, so this is the linkcell path.
        let xyz: Vec<[f64; 3]> = (0..125)
            .map(|s| [s / 25, s / 5 % 5, s % 5].map(|c| (c as f64 + 0.5) * 5.0))
            .collect();
        let cell = [25.0, 0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 25.0];
        let rows = [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]];
        let mut count = 0;
        assert_eq!(fast_pairs(&xyz, &cell, 7.0, &mut [], &mut count), 1);
        assert!(count > 0 && count < xyz.len() * (xyz.len() - 1) / 2);
        let mut ij = vec![[-1; 2]; count];
        let mut filled = 0;
        assert_eq!(fast_pairs(&xyz, &cell, 7.0, &mut ij, &mut filled), 0);
        assert_eq!(filled, count);
        let got: Vec<(i32, i32)> = ij.iter().map(|&[i, j]| (i, j)).collect();
        assert_eq!(got, pairs_of(&xyz, &rows, [true, true, true], 7.0));
    }
}
