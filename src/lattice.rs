//! Candidate structures grown from local order, rather than reached by hopping.
//!
//! Everything else in this crate decides where a chain hops next. None of it
//! changes what a hop can reach, because reachability is set by the move, and
//! the move is a displacement followed by a quench. Measured on 98 points, 12
//! million charged evaluations buy 401 177 hops over 11 366 distinct basins,
//! about 35 visits each, on a landscape whose minimum count grows like
//! `e^{alpha N}`. Filling visited basins pushes the chain into neighbouring
//! unvisited basins in the same funnel, of which there are effectively
//! unboundedly many, and never to the funnel boundary. Cameron reads the same
//! thing off the spectrum: the icosahedral escape mode on 38 points sits around
//! the 245th, with no gap to exploit.
//!
//! A structure whose local order differs everywhere is a different funnel, and
//! local order is something a structure can be *built* to have. Points are
//! indistinguishable, so a set of positions is a complete proposal: growing one
//! and quenching into it crosses a funnel boundary in a single step.
//!
//! # Nothing here is specific to one potential
//!
//! The order to grow is read off the structure the chain is standing on, by
//! taking the neighbour offsets of its best-coordinated point. That works
//! wherever a neighbour shell means anything, needs no template library, and
//! carries no assumption about what is being optimised: at a relaxed
//! Lennard-Jones geometry it reads a close-packed shell, at a relaxed Morse
//! geometry it reads whatever that potential prefers, and at a molecular
//! geometry it reads that.
//!
//! [`Source::Named`] offers the alternative orders instead, from the
//! classifier's own template library in [`crate::structure`], so a chain sitting
//! in close packing can be handed icosahedral order and the other way round.
//! Which is worth proposing is left to the allocator; nothing here knows that
//! 38 points want a truncated octahedron.
//!
//! The same construction is what a ring or cage topology would drive for a
//! molecular system, where the repeated unit is a cage rather than a
//! coordination shell. The growth below takes offsets and does not ask where
//! they came from.

use crate::structure::Template;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use std::f64::consts::PI;

/// Where the local order to grow comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Read off the current structure's best-coordinated point.
    ///
    /// The general case, and the one that assumes nothing.
    Observed,
    /// One of the classifier's ideal local environments.
    Named(Template),
    /// Fivefold bipyramid, the local order of a decahedral packing.
    ///
    /// It does not tile space. Growth accumulates the strain a decahedron
    /// carries, and the quench decides the minimum, the same way an
    /// icosahedral shell does.
    Pentagonal,
}

impl Source {
    /// The sources worth offering as separate arms.
    pub fn library() -> Vec<Source> {
        vec![
            Source::Observed,
            Source::Named(Template::FaceCentredCubic),
            Source::Named(Template::HexagonalClosePacked),
            Source::Named(Template::Icosahedral),
            Source::Named(Template::SimpleCubic),
        ]
    }

    /// Short name for reporting.
    pub fn name(&self) -> &'static str {
        match self {
            Source::Observed => "observed",
            Source::Named(Template::FaceCentredCubic) => "fcc",
            Source::Named(Template::HexagonalClosePacked) => "hcp",
            Source::Named(Template::Icosahedral) => "ico",
            Source::Named(Template::SimpleCubic) => "sc",
            Source::Named(Template::Other) => "other",
            Source::Pentagonal => "pentagonal",
        }
    }
}

/// `n` sites of a pentagonal column, scaled so the nearest neighbour is `scale`.
///
/// Shells grow on a fivefold axis. The sites closest to a small random centre
/// are kept, then each coordinate is shifted by a fraction of `scale`.
pub fn decahedral_cut<R: Rng + ?Sized>(n: usize, scale: f64, rng: &mut R) -> Array1<f64> {
    if n == 0 {
        return Array1::zeros(0);
    }
    let radius_unit = 1.0 / (2.0 * (PI / 5.0).sin());
    let mut sites: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
    for shell in 1i32..10 {
        if sites.len() >= n + 40 {
            break;
        }
        for z in -shell..=shell {
            let m = shell - z.unsigned_abs() as i32;
            if m <= 0 {
                let point = [0.0, 0.0, z as f64];
                if sites.iter().all(|old| sq(old, &point) > 0.04) {
                    sites.push(point);
                }
                continue;
            }
            let rad = m as f64 * radius_unit;
            for k in 0..5 {
                let angle = 2.0 * PI * (k as f64) / 5.0;
                let point = [rad * angle.cos(), rad * angle.sin(), z as f64];
                if sites.iter().all(|old| sq(old, &point) > 0.04) {
                    sites.push(point);
                }
            }
        }
    }
    let centre = [
        0.35 * (rng.random::<f64>() - 0.5),
        0.35 * (rng.random::<f64>() - 0.5),
        0.35 * (rng.random::<f64>() - 0.5),
    ];
    sites.sort_by(|left, right| {
        sq(left, &centre)
            .partial_cmp(&sq(right, &centre))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sites.truncate(n.min(sites.len()));
    let mut raw = Array1::zeros(3 * sites.len());
    for (i, site) in sites.iter().enumerate() {
        for k in 0..3 {
            raw[3 * i + k] = site[k];
        }
    }
    let local = nearest_neighbour_scale(raw.view()).max(1.0e-6);
    let factor = scale / local;
    for value in raw.iter_mut() {
        *value = *value * factor + 0.02 * scale * (rng.random::<f64>() - 0.5);
    }
    raw
}

/// Neighbour offsets of a centred pentagonal bipyramid.
///
/// Ring-ring and apex-ring distances are one. The centre-ring distance is
/// shorter, which is the strain of a fivefold axis rather than a fitted
/// global minimum.
fn pentagonal_shell() -> Vec<[f64; 3]> {
    let radius = 1.0 / (2.0 * (PI / 5.0).sin());
    let height = (1.0 - radius * radius).max(0.05).sqrt();
    let mut shell = Vec::with_capacity(7);
    for k in 0..5 {
        let angle = 2.0 * PI * (k as f64) / 5.0;
        shell.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
    }
    shell.push([0.0, 0.0, height]);
    shell.push([0.0, 0.0, -height]);
    shell
}

/// Mean distance to the nearest other point.
///
/// The length scale a template is built at, taken from the structure so the
/// move carries no knowledge of the potential.
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

/// Neighbour offsets of the best-coordinated point, in units of the nearest
/// neighbour distance.
///
/// The best-coordinated point rather than an average one because that is where
/// the structure's order is expressed most completely: a surface point's shell
/// is a fragment, and propagating a fragment grows a fragment.
pub fn observed_order(x: ArrayView1<f64>, cutoff: f64) -> Vec<[f64; 3]> {
    observed_order_at(x, cutoff, nearest_neighbour_scale(x))
}

fn observed_order_at(x: ArrayView1<f64>, cutoff: f64, scale: f64) -> Vec<[f64; 3]> {
    let n = x.len() / 3;
    if n < 2 {
        return Vec::new();
    }
    let reach = cutoff * scale;
    let mut best = (0usize, 0usize);
    let mut shells: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut shell = Vec::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = [
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r <= reach {
                shell.push([d[0] / scale, d[1] / scale, d[2] / scale]);
            }
        }
        if shell.len() > best.1 {
            best = (i, shell.len());
        }
        shells.push(shell);
    }
    shells.swap_remove(best.0)
}

/// Grows `n` sites by repeatedly placing `offsets` around already placed ones.
///
/// Breadth first, so the structure grows outward from the seed and stays
/// compact, which is what a cluster minimum is. A candidate site closer than
/// half a spacing to a placed one is the same site and is dropped.
///
/// For an order that tiles space this reproduces its lattice. For one that does
/// not, icosahedral being the case that matters here, the propagation cannot
/// close and the growth accumulates strain, which is the physical situation
/// rather than a defect of the method: that is why icosahedral clusters stop
/// competing above a few hundred points.
pub fn grow(offsets: &[[f64; 3]], want: usize) -> Vec<[f64; 3]> {
    grow_from(&[], offsets, want)
}

/// As [`grow`], starting from `seed` rather than from a single point.
///
/// A seed of existing positions is what lets a proposal keep part of the
/// structure it came from: the new order grows around the kept part in the
/// growth's own orientation, so the two meet at an interface instead of one
/// replacing the other.
pub fn grow_from(seed: &[[f64; 3]], offsets: &[[f64; 3]], want: usize) -> Vec<[f64; 3]> {
    if offsets.is_empty() {
        return Vec::new();
    }
    let start = if seed.is_empty() {
        vec![[0.0, 0.0, 0.0]]
    } else {
        seed.to_vec()
    };
    let mut sites = Cells::new(start);
    let mut head = 0usize;
    while sites.len() < want && head < sites.len() {
        let s = sites.points[head];
        head += 1;
        for o in offsets {
            let c = [s[0] + o[0], s[1] + o[1], s[2] + o[2]];
            if !sites.any_within(&c, SAME_SITE, |_| true) {
                sites.push(c);
                if sites.len() >= want {
                    break;
                }
            }
        }
    }
    sites.points
}

/// Squared distance below which a grown site is one already placed.
const SAME_SITE: f64 = 0.25;

/// Squared distance below which two grown sites are neighbours.
const BONDED: f64 = 1.44;

/// Largest coordinate whose cell index converts to and from `f64` exactly.
const CELL_RANGE: f64 = (1u64 << 52) as f64;

fn sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Points binned in unit cubes, the length the growth frame is measured in.
///
/// A cell index times the unit is exact, so a point lies in cell `k` exactly
/// when `k <= p < k + 1`. A query widens its range one cell at a time while
/// the next face is nearer than the cutoff under the same rounded square
/// that [`sq`] takes, so every cell it leaves out is already too far for
/// any point inside it. It therefore applies the all-pairs test to every
/// point that test could accept, for any cutoff.
///
/// Bins are kept only while they cost no more than one all-pairs pass over
/// the points they hold, and only for finite coordinates in exact index
/// range. Otherwise the table scans every point, which is the same answer.
struct Cells {
    points: Vec<[f64; 3]>,
    next: Vec<usize>,
    lo: [i64; 3],
    dim: [i64; 3],
    head: Vec<usize>,
    binned: bool,
    retry_at: usize,
}

impl Cells {
    fn new(points: Vec<[f64; 3]>) -> Self {
        let mut cells = Self {
            points,
            next: Vec::new(),
            lo: [0; 3],
            dim: [0; 3],
            head: Vec::new(),
            binned: false,
            retry_at: 0,
        };
        cells.rebin();
        cells
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn holds(p: &[f64; 3]) -> bool {
        p.iter().all(|v| v.abs() < CELL_RANGE)
    }

    fn cell(p: &[f64; 3]) -> [i64; 3] {
        let mut c = [0i64; 3];
        for k in 0..3 {
            let mut i = p[k] as i64;
            while i as f64 > p[k] {
                i -= 1;
            }
            while (i + 1) as f64 <= p[k] {
                i += 1;
            }
            c[k] = i;
        }
        c
    }

    fn slot(&self, c: [i64; 3]) -> Option<usize> {
        let mut s = 0i64;
        for k in 0..3 {
            let o = c[k] - self.lo[k];
            if o < 0 || o >= self.dim[k] {
                return None;
            }
            s = s * self.dim[k] + o;
        }
        Some(s as usize)
    }

    /// Bins every point, with as much room again on each axis to grow into.
    fn rebin(&mut self) {
        self.binned = false;
        self.retry_at = 2 * self.points.len().max(1);
        self.head.clear();
        self.next.clear();
        if !self.points.iter().all(Self::holds) {
            return;
        }
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for (id, p) in self.points.iter().enumerate() {
            let c = Self::cell(p);
            for k in 0..3 {
                if id == 0 || c[k] < lo[k] {
                    lo[k] = c[k];
                }
                if id == 0 || c[k] > hi[k] {
                    hi[k] = c[k];
                }
            }
        }
        let budget = (self.points.len() as u128).pow(2);
        let mut total = 1u128;
        for k in 0..3 {
            let span = hi[k] - lo[k] + 1;
            self.lo[k] = lo[k] - span / 2 - 1;
            self.dim[k] = 2 * span + 2;
            total *= self.dim[k] as u128;
        }
        if total > budget {
            return;
        }
        self.head.resize(total as usize, usize::MAX);
        self.next.resize(self.points.len(), usize::MAX);
        for id in 0..self.points.len() {
            let s = self
                .slot(Self::cell(&self.points[id]))
                .expect("bounds cover every point");
            self.next[id] = self.head[s];
            self.head[s] = id;
        }
        self.binned = true;
    }

    fn push(&mut self, p: [f64; 3]) {
        self.points.push(p);
        if !self.binned {
            if self.points.len() >= self.retry_at {
                self.rebin();
            }
            return;
        }
        let id = self.points.len() - 1;
        self.next.push(usize::MAX);
        match Self::holds(&p).then(|| self.slot(Self::cell(&p))).flatten() {
            Some(s) => {
                self.next[id] = self.head[s];
                self.head[s] = id;
            }
            None => self.rebin(),
        }
    }

    /// Whether `hit` accepts some point with `sq < limit` from `c`, offering
    /// it each such point until it does.
    fn any_within(&self, c: &[f64; 3], limit: f64, mut hit: impl FnMut(usize) -> bool) -> bool {
        if !(self.binned && Self::holds(c)) {
            return (0..self.points.len()).any(|id| sq(&self.points[id], c) < limit && hit(id));
        }
        let home = Self::cell(c);
        let mut from = home;
        let mut to = home;
        for k in 0..3 {
            while from[k] > self.lo[k] && (c[k] - from[k] as f64).powi(2) < limit {
                from[k] -= 1;
            }
            while to[k] < self.lo[k] + self.dim[k] - 1
                && ((to[k] + 1) as f64 - c[k]).powi(2) < limit
            {
                to[k] += 1;
            }
        }
        for a in from[0]..=to[0] {
            for b in from[1]..=to[1] {
                for z in from[2]..=to[2] {
                    let Some(s) = self.slot([a, b, z]) else {
                        continue;
                    };
                    let mut id = self.head[s];
                    while id != usize::MAX {
                        if sq(&self.points[id], c) < limit && hit(id) {
                            return true;
                        }
                        id = self.next[id];
                    }
                }
            }
        }
        false
    }
}

/// Removes the least-coordinated site until `n` remain, returning survivors.
///
/// Ties go to the site furthest from the centre, then to the lowest index.
/// The jitter in the centre is what makes repeated draws from one source
/// different structures: a body peeled about a site and one peeled about a
/// hole are different clusters, and both are minima of the packing at some
/// size. The buckets remove exactly the site a scan of every survivor picks;
/// a NaN distance, which no order agrees with that scan on, takes the scan.
fn peel(adj: &[Vec<usize>], centre_distance: &[f64], n: usize) -> Vec<bool> {
    let m = adj.len();
    let mut count: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    let mut alive = vec![true; m];
    let mut live = m;
    if centre_distance.iter().any(|d| d.is_nan()) {
        while live > n {
            let mut worst = usize::MAX;
            for i in 0..m {
                if !alive[i] {
                    continue;
                }
                if worst == usize::MAX
                    || count[i] < count[worst]
                    || (count[i] == count[worst] && centre_distance[i] > centre_distance[worst])
                {
                    worst = i;
                }
            }
            alive[worst] = false;
            live -= 1;
            for &j in &adj[worst] {
                if alive[j] {
                    count[j] -= 1;
                }
            }
        }
        return alive;
    }
    // Survivors by neighbour count. Only the lowest occupied count is ever
    // searched, and a removal lowers its neighbours by one each.
    let top = count.iter().copied().max().unwrap_or(0);
    let mut bucket: Vec<Vec<usize>> = vec![Vec::new(); top + 1];
    let mut slot = vec![0usize; m];
    for i in 0..m {
        slot[i] = bucket[count[i]].len();
        bucket[count[i]].push(i);
    }
    fn take(bucket: &mut [Vec<usize>], slot: &mut [usize], c: usize, i: usize) {
        let at = slot[i];
        bucket[c].swap_remove(at);
        if let Some(&moved) = bucket[c].get(at) {
            slot[moved] = at;
        }
    }
    let mut low = 0usize;
    while live > n {
        while bucket[low].is_empty() {
            low += 1;
        }
        let mut worst = bucket[low][0];
        for &i in &bucket[low][1..] {
            let (d, w) = (centre_distance[i], centre_distance[worst]);
            if d > w || (d == w && i < worst) {
                worst = i;
            }
        }
        take(&mut bucket, &mut slot, low, worst);
        alive[worst] = false;
        live -= 1;
        for &j in &adj[worst] {
            if alive[j] {
                take(&mut bucket, &mut slot, count[j], j);
                count[j] -= 1;
                slot[j] = bucket[count[j]].len();
                bucket[count[j]].push(j);
                low = low.min(count[j]);
            }
        }
    }
    alive
}

/// Neighbours of every site, `1e-9 < sq < BONDED`, in no particular order.
///
/// The sites are sorted by unit cell, so a row of cells along the last axis is
/// one contiguous run, and each site's search covers the rows the face test
/// of [`Cells`] admits. Unbinnable input takes the all-pairs loop.
fn bonded_neighbours(sites: &[[f64; 3]]) -> Vec<Vec<usize>> {
    let m = sites.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    let link = |adj: &mut Vec<Vec<usize>>, i: usize, j: usize| {
        let d = sq(&sites[i], &sites[j]);
        if d < BONDED && d > 1e-9 {
            adj[i].push(j);
            adj[j].push(i);
        }
    };
    let cells: Vec<[i64; 3]> = sites.iter().map(Cells::cell).collect();
    let mut lo = [0i64; 3];
    let mut dim = [0i64; 3];
    let mut total = 0u128;
    if m > 0 && sites.iter().all(Cells::holds) {
        total = 1;
        for k in 0..3 {
            lo[k] = cells.iter().map(|c| c[k]).min().unwrap_or(0);
            dim[k] = cells.iter().map(|c| c[k]).max().unwrap_or(0) - lo[k] + 1;
            total *= dim[k] as u128;
        }
    }
    if total == 0 || total > (m as u128).pow(2) {
        for i in 0..m {
            for j in (i + 1)..m {
                link(&mut adj, i, j);
            }
        }
        return adj;
    }
    let slot = |c: [i64; 3]| {
        (((c[0] - lo[0]) * dim[1] + (c[1] - lo[1])) * dim[2] + (c[2] - lo[2])) as usize
    };
    let mut start = vec![0usize; total as usize + 1];
    for c in &cells {
        start[slot(*c) + 1] += 1;
    }
    for s in 0..total as usize {
        start[s + 1] += start[s];
    }
    let mut fill = start.clone();
    let mut order = vec![0usize; m];
    for (i, c) in cells.iter().enumerate() {
        let s = slot(*c);
        order[fill[s]] = i;
        fill[s] += 1;
    }
    for i in 0..m {
        let (p, home) = (&sites[i], cells[i]);
        let mut from = home;
        let mut to = home;
        for k in 0..3 {
            while from[k] > lo[k] && (p[k] - from[k] as f64).powi(2) < BONDED {
                from[k] -= 1;
            }
            while to[k] < lo[k] + dim[k] - 1 && ((to[k] + 1) as f64 - p[k]).powi(2) < BONDED {
                to[k] += 1;
            }
        }
        for a in from[0]..=to[0] {
            for b in from[1]..=to[1] {
                let first = start[slot([a, b, from[2]])];
                let last = start[slot([a, b, to[2]]) + 1];
                for &j in &order[first..last] {
                    if j > i {
                        link(&mut adj, i, j);
                    }
                }
            }
        }
    }
    adj
}

/// Builds an `n`-point candidate carrying `source`'s local order at the length
/// scale of `x`.
///
/// Grown wide and then cut to the `n` sites nearest a jittered centre, which is
/// what makes the candidate compact. Which polyhedron the cut produces is left
/// to the quench rather than imposed, and the jitter means repeated proposals
/// from one source are different structures: a close-packed lattice cut about a
/// site gives a different cluster from one cut about a hole, and both are
/// minima of that packing at some size.
pub fn candidate<R: Rng + ?Sized>(
    source: Source,
    x: ArrayView1<f64>,
    n: usize,
    rng: &mut R,
) -> Array1<f64> {
    candidate_keeping(source, x, n, 0.0, rng)
}

/// As [`candidate`], retaining a `keep` fraction of the current structure as
/// the seed the new order grows from.
///
/// The knob that makes this one move rather than two. At `keep = 0` the
/// structure is discarded and a body is grown from nothing, which is the
/// largest step the move set has. At `keep = 1` nothing is regrown. In between,
/// the best-coordinated part of the current structure is kept and the new order
/// is grown around it in its own orientation, so the proposal carries an
/// interface between the two: a twin or a stacking fault, which is how a
/// morphology actually changes rather than how it is replaced.
///
/// Which value pays is not fixed here. It is a continuous parameter, so it is
/// something a posterior can be held over, which is what [`crate::construct`]
/// does.
pub fn candidate_keeping<R: Rng + ?Sized>(
    source: Source,
    x: ArrayView1<f64>,
    n: usize,
    keep: f64,
    rng: &mut R,
) -> Array1<f64> {
    match grow_body(source, x, n, keep) {
        Some(body) => cut(&body, n, rng),
        None => x.to_owned(),
    }
}

/// As [`candidate`], keeping the growth in `growth` across calls.
///
/// The growth depends on the structure alone and the cut on the random draws
/// alone, so a call on the same structure, bit for bit, reuses the growth and
/// draws exactly what [`candidate`] draws. A stalled chain proposes from one
/// structure many times, which is when this move is drawn.
pub fn candidate_reusing<R: Rng + ?Sized>(
    growth: &mut Growth,
    source: Source,
    x: ArrayView1<f64>,
    n: usize,
    rng: &mut R,
) -> Array1<f64> {
    let bits: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
    let same = growth
        .key
        .as_ref()
        .is_some_and(|(held, held_n, held_bits)| {
            *held == source && *held_n == n && *held_bits == bits
        });
    if !same {
        growth.body = grow_body(source, x, n, 0.0);
        growth.key = Some((source, n, bits));
    }
    match &growth.body {
        Some(body) => cut(body, n, rng),
        None => x.to_owned(),
    }
}

/// The part of a candidate that the random draws do not touch.
#[derive(Debug, Clone)]
struct Body {
    scale: f64,
    sites: Vec<[f64; 3]>,
    adj: Vec<Vec<usize>>,
}

/// A growth held between calls of [`candidate_reusing`].
#[derive(Debug, Clone, Default)]
pub struct Growth {
    key: Option<(Source, usize, Vec<u64>)>,
    body: Option<Body>,
}

/// Grows the wide body a candidate is cut from, or `None` when the candidate
/// is the input itself.
fn grow_body(source: Source, x: ArrayView1<f64>, n: usize, keep: f64) -> Option<Body> {
    let scale = nearest_neighbour_scale(x);
    let offsets = match source {
        Source::Observed => observed_order_at(x, 1.35, scale),
        Source::Named(t) => t.points(),
        Source::Pentagonal => pentagonal_shell(),
    };
    if offsets.is_empty() || n == 0 {
        return None;
    }

    // The kept core, in units of the spacing and centred, which is the frame
    // the growth works in.
    let n_parent = x.len() / 3;
    let n_keep = ((keep.clamp(0.0, 1.0) * n as f64).round() as usize).min(n_parent.min(n));
    let mut seed: Vec<[f64; 3]> = Vec::new();
    if n_keep > 0 {
        let mut centre = [0.0; 3];
        for i in 0..n_parent {
            for k in 0..3 {
                centre[k] += x[3 * i + k];
            }
        }
        for c in centre.iter_mut() {
            *c /= n_parent as f64;
        }
        let pts: Vec<[f64; 3]> = (0..n_parent)
            .map(|i| {
                [
                    (x[3 * i] - centre[0]) / scale,
                    (x[3 * i + 1] - centre[1]) / scale,
                    (x[3 * i + 2] - centre[2]) / scale,
                ]
            })
            .collect();
        let counts: Vec<usize> = pts
            .iter()
            .map(|a| {
                pts.iter()
                    .filter(|b| sq(a, b) < 1.44 && sq(a, b) > 1e-9)
                    .count()
            })
            .collect();
        let mut order: Vec<usize> = (0..n_parent).collect();
        order.sort_by(|&i, &j| counts[j].cmp(&counts[i]));
        order.truncate(n_keep);
        seed = order.into_iter().map(|i| pts[i]).collect();
    }

    let sites = grow_from(&seed, &offsets, (n * 6).max(n + 32));
    if sites.len() < n {
        return None;
    }
    let adj = bonded_neighbours(&sites);
    Some(Body { scale, sites, adj })
}

/// Peels `body` to `n` points about a jittered centre, then turns and jitters
/// the result.
fn cut<R: Rng + ?Sized>(body: &Body, n: usize, rng: &mut R) -> Array1<f64> {
    let Body { scale, sites, adj } = body;
    let scale = *scale;
    let _centre = [
        rng.random::<f64>() - 0.5,
        rng.random::<f64>() - 0.5,
        rng.random::<f64>() - 0.5,
    ];
    // Peel to the compact body, rather than cutting a sphere.
    //
    // Taking the n sites nearest a centre cuts a sphere out of the packing, and
    // cluster minima are not spheres: they are polyhedra whose faces are the
    // packing's close-packed planes, which is what a truncated octahedron or a
    // Marks decahedron is. Measured on 98 points, the spherical cut proposed
    // structures quenching to -535.4 where the displacement moves already stood
    // at -538.7, so the move fired, was accepted at one draw in three, and led
    // nowhere.
    //
    // Ranking the grown sites by coordination once does not fix it and was
    // measured not to: mean coordination 7.82 either way. A breadth-first
    // growth is already roughly a ball, so its best-coordinated sites are its
    // interior, which is the same ball.
    //
    // Peeling is the construction that works. Delete the least-coordinated site,
    // recount its neighbours, repeat until n remain. Facets appear because
    // removing an atom from a partly filled plane lowers its neighbours'
    // coordination and makes them the next to go, so the plane empties before
    // any full plane is touched. That is the Wulff construction with every
    // facet weighted equally, and it needs nothing but neighbour counting, so
    // it stays as general as the growth it cuts.
    let m = sites.len();
    let centre = [
        rng.random::<f64>() - 0.5,
        rng.random::<f64>() - 0.5,
        rng.random::<f64>() - 0.5,
    ];
    // The centre is fixed for the peel, so each site's squared distance is
    // one number. Recomputing it on every comparison picked the same site.
    let centre_distance: Vec<f64> = sites.iter().map(|site| sq(site, &centre)).collect();
    let alive = peel(adj, &centre_distance, n);
    let sites: Vec<[f64; 3]> = (0..m).filter(|&i| alive[i]).map(|i| sites[i]).collect();

    let rot = random_rotation(rng);
    let mut out = Array1::zeros(3 * n);
    for (i, s) in sites.iter().enumerate() {
        // Jitter well below the spacing: an exact template sits on a symmetric
        // saddle often enough that a quench from it goes nowhere.
        for k in 0..3 {
            let v: f64 = (0..3).map(|m| rot[k][m] * s[m]).sum();
            out[3 * i + k] = scale * v + 0.02 * (rng.random::<f64>() - 0.5);
        }
    }
    out
}

fn random_rotation<R: Rng + ?Sized>(rng: &mut R) -> [[f64; 3]; 3] {
    let (u1, u2, u3) = (
        rng.random::<f64>(),
        rng.random::<f64>(),
        rng.random::<f64>(),
    );
    let (x, y, z, w) = (
        (1.0 - u1).sqrt() * (2.0 * PI * u2).sin(),
        (1.0 - u1).sqrt() * (2.0 * PI * u2).cos(),
        u1.sqrt() * (2.0 * PI * u3).sin(),
        u1.sqrt() * (2.0 * PI * u3).cos(),
    );
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(11)
    }

    /// A relaxed close-packed fragment, built by growing the ideal shell, which
    /// is the input a reseed would see mid-run.
    fn packed(n: usize) -> Array1<f64> {
        let sites = grow(&Template::FaceCentredCubic.points(), n);
        let mut x = Array1::zeros(3 * n);
        for (i, s) in sites.iter().take(n).enumerate() {
            for k in 0..3 {
                x[3 * i + k] = 1.12 * s[k];
            }
        }
        x
    }

    /// Every source has to produce exactly the requested count, or the move
    /// hands the objective a different problem.
    #[test]
    fn every_source_produces_the_requested_count() {
        let mut r = rng();
        for s in Source::library() {
            for n in [13, 38, 55, 75, 98] {
                let x = packed(n);
                let c = candidate(s, x.view(), n, &mut r);
                assert_eq!(c.len(), 3 * n, "{} at n={n}", s.name());
            }
        }
    }

    /// The candidate has to come out at the length scale of the structure it
    /// was asked about, since that is the only thing tying it to the potential.
    #[test]
    fn the_scale_follows_the_structure_not_a_constant() {
        let mut r = rng();
        let mut x = packed(55);
        let c1 = candidate(Source::Observed, x.view(), 55, &mut r);
        x *= 2.0;
        let c2 = candidate(Source::Observed, x.view(), 55, &mut r);
        let s1 = nearest_neighbour_scale(c1.view());
        let s2 = nearest_neighbour_scale(c2.view());
        assert!(
            (s2 / s1 - 2.0).abs() < 0.2,
            "scales {s1} and {s2} are not a factor of two apart"
        );
    }

    /// No two points may coincide, or the potential overflows and the
    /// relaxation counts a failure instead of work.
    #[test]
    fn no_source_produces_coincident_points() {
        let mut r = rng();
        for s in Source::library() {
            for n in [38, 98] {
                let x = packed(n);
                let c = candidate(s, x.view(), n, &mut r);
                for i in 0..n {
                    for j in (i + 1)..n {
                        let d: f64 = (0..3)
                            .map(|k| {
                                let v = c[3 * i + k] - c[3 * j + k];
                                v * v
                            })
                            .sum::<f64>()
                            .sqrt();
                        assert!(d > 0.4, "{} n={n}: points {i},{j} at {d}", s.name());
                    }
                }
            }
        }
    }

    /// The peel has to produce a more compact body than a spherical cut of the
    /// same grown set, which is the claim it is there for. Compared directly
    /// rather than against a number, since the number depends on the packing.
    #[test]
    fn peeling_beats_a_spherical_cut() {
        let mut r = rng();
        let x = packed(120);
        let peeled = candidate(
            Source::Named(Template::FaceCentredCubic),
            x.view(),
            55,
            &mut r,
        );
        // The same grown set, cut by radius, which is what this replaced.
        let grown = grow(&Template::FaceCentredCubic.points(), 55 * 6);
        let mut by_radius = grown.clone();
        by_radius.sort_by(|a, b| {
            sq(a, &[0.0, 0.0, 0.0])
                .partial_cmp(&sq(b, &[0.0, 0.0, 0.0]))
                .unwrap()
        });
        by_radius.truncate(55);
        let sphere: Array1<f64> = {
            let mut v = Array1::zeros(3 * 55);
            for (i, s) in by_radius.iter().enumerate() {
                for k in 0..3 {
                    v[3 * i + k] = 1.12 * s[k];
                }
            }
            v
        };
        let a = mean_coordination(peeled.view(), 55);
        let b = mean_coordination(sphere.view(), 55);
        assert!(
            a > b,
            "peeled body averages {a} neighbours, spherical cut {b}"
        );
    }

    fn mean_coordination(c: ArrayView1<f64>, n: usize) -> f64 {
        let scale = nearest_neighbour_scale(c);
        let mut total = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d: f64 = (0..3)
                    .map(|k| {
                        let v = c[3 * i + k] - c[3 * j + k];
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt();
                if d < 1.2 * scale {
                    total += 1;
                }
            }
        }
        total as f64 / n as f64
    }

    #[allow(dead_code)]
    fn unused_old_compactness_check() {
        let mut r = rng();
        let x = packed(120);
        let c = candidate(
            Source::Named(Template::FaceCentredCubic),
            x.view(),
            55,
            &mut r,
        );
        let scale = nearest_neighbour_scale(c.view());
        let mut total = 0usize;
        for i in 0..55 {
            for j in 0..55 {
                if i == j {
                    continue;
                }
                let d: f64 = (0..3)
                    .map(|k| {
                        let v = c[3 * i + k] - c[3 * j + k];
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt();
                if d < 1.2 * scale {
                    total += 1;
                }
            }
        }
        let mean = total as f64 / 55.0;
        // A 55-point spherical cut of close packing averages near 7.6
        // neighbours; a compact polyhedral one is above 8.
        assert!(mean > 8.0, "mean coordination {mean} is not compact");
    }

    /// The whole point of the move: the named orders have to be different
    /// structures from each other, or it offers several names for one proposal.
    /// Compared by the sorted distance spectrum, the crate's own basin
    /// descriptor.
    #[test]
    fn the_named_orders_are_different_structures() {
        let mut r = rng();
        let x = packed(55);
        let spectra: Vec<Vec<f64>> = [
            Template::FaceCentredCubic,
            Template::Icosahedral,
            Template::SimpleCubic,
        ]
        .iter()
        .map(|t| {
            spectrum(
                candidate(Source::Named(*t), x.view(), 55, &mut r).view(),
                55,
            )
        })
        .collect();
        for i in 0..spectra.len() {
            for j in (i + 1)..spectra.len() {
                let diff: f64 = spectra[i]
                    .iter()
                    .zip(spectra[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                assert!(diff > 1.0, "orders {i} and {j} differ by only {diff}");
            }
        }
    }

    fn spectrum(c: ArrayView1<f64>, n: usize) -> Vec<f64> {
        let mut d = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                d.push(
                    (0..3)
                        .map(|k| {
                            let v = c[3 * i + k] - c[3 * j + k];
                            v * v
                        })
                        .sum::<f64>()
                        .sqrt(),
                );
            }
        }
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d
    }

    /// Reading the order off the structure has to recover the order the
    /// structure has, or the general path is not general, it is arbitrary.
    #[test]
    fn observed_order_recovers_the_shell_it_was_built_from() {
        let x = packed(80);
        let o = observed_order(x.view(), 1.35);
        assert_eq!(o.len(), 12, "close packing has twelve nearest neighbours");
        for p in &o {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 0.1, "offset at radius {r}, wanted 1");
        }
    }

    /// And growing what was read has to give back the same packing, which is
    /// the property that lets the move work without a template library.
    #[test]
    fn growing_the_observed_order_reproduces_the_packing() {
        let mut r = rng();
        let x = packed(80);
        let c = candidate(Source::Observed, x.view(), 55, &mut r);
        let named = candidate(
            Source::Named(Template::FaceCentredCubic),
            x.view(),
            55,
            &mut r,
        );
        let a = spectrum(c.view(), 55);
        let b = spectrum(named.view(), 55);
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 2.0,
            "observed order gave a structure {diff} from the packing it was read from"
        );
    }

    /// The all-pairs growth the binned one has to reproduce.
    fn grow_by_scan(seed: &[[f64; 3]], offsets: &[[f64; 3]], want: usize) -> Vec<[f64; 3]> {
        if offsets.is_empty() {
            return Vec::new();
        }
        let mut sites: Vec<[f64; 3]> = if seed.is_empty() {
            vec![[0.0, 0.0, 0.0]]
        } else {
            seed.to_vec()
        };
        let mut head = 0usize;
        while sites.len() < want && head < sites.len() {
            let s = sites[head];
            head += 1;
            for o in offsets {
                let c = [s[0] + o[0], s[1] + o[1], s[2] + o[2]];
                if !sites.iter().any(|t| sq(t, &c) < 0.25) {
                    sites.push(c);
                    if sites.len() >= want {
                        break;
                    }
                }
            }
        }
        sites
    }

    /// The all-pairs peel the bucketed one has to reproduce.
    fn peel_by_scan(sites: &[[f64; 3]], centre: [f64; 3], n: usize) -> Vec<bool> {
        let m = sites.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
        for i in 0..m {
            for j in (i + 1)..m {
                let d = sq(&sites[i], &sites[j]);
                if d < 1.44 && d > 1e-9 {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }
        let mut count: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let mut alive = vec![true; m];
        let centre_distance: Vec<f64> = sites.iter().map(|site| sq(site, &centre)).collect();
        let mut live = m;
        while live > n {
            let mut worst = usize::MAX;
            for i in 0..m {
                if !alive[i] {
                    continue;
                }
                if worst == usize::MAX
                    || count[i] < count[worst]
                    || (count[i] == count[worst] && centre_distance[i] > centre_distance[worst])
                {
                    worst = i;
                }
            }
            alive[worst] = false;
            live -= 1;
            for &j in &adj[worst] {
                if alive[j] {
                    count[j] -= 1;
                }
            }
        }
        alive
    }

    fn bits(p: &[[f64; 3]]) -> Vec<[u64; 3]> {
        p.iter().map(|q| q.map(f64::to_bits)).collect()
    }

    /// Shells read off perturbed packings, ideal shells, and shells with
    /// offsets near the duplicate radius, which is where a cell boundary
    /// would show if the binned test missed a neighbour.
    fn shells(r: &mut StdRng) -> Vec<Vec<[f64; 3]>> {
        let mut out = vec![
            Template::FaceCentredCubic.points(),
            Template::Icosahedral.points(),
            pentagonal_shell(),
        ];
        for n in [13usize, 38, 55, 80] {
            let mut x = packed(n);
            for v in x.iter_mut() {
                *v += 0.15 * (r.random::<f64>() - 0.5);
            }
            out.push(observed_order(x.view(), 1.35));
        }
        let near: Vec<[f64; 3]> = (0..10)
            .map(|_| {
                let u = [
                    r.random::<f64>() - 0.5,
                    r.random::<f64>() - 0.5,
                    r.random::<f64>() - 0.5,
                ];
                let l = sq(&u, &[0.0; 3]).sqrt().max(1e-6);
                let len = 0.5 + 0.01 * (r.random::<f64>() - 0.5);
                [u[0] / l * len, u[1] / l * len, u[2] / l * len]
            })
            .collect();
        out.push(near);
        out
    }

    #[test]
    fn binned_growth_reproduces_the_all_pairs_growth_bit_for_bit() {
        let mut r = rng();
        for offsets in shells(&mut r) {
            for want in [1usize, 2, 17, 228, 480, 900] {
                let a = grow_from(&[], &offsets, want);
                let b = grow_by_scan(&[], &offsets, want);
                assert_eq!(bits(&a), bits(&b), "want={want}");
                let seed: Vec<[f64; 3]> = a
                    .iter()
                    .take(7)
                    .map(|p| [p[0] + 0.3, p[1], p[2] - 0.2])
                    .collect();
                assert_eq!(
                    bits(&grow_from(&seed, &offsets, want)),
                    bits(&grow_by_scan(&seed, &offsets, want))
                );
            }
        }
    }

    #[test]
    fn unbinnable_growth_reproduces_the_all_pairs_growth() {
        let far = [
            [1e300, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [f64::NAN, 0.0, 0.0],
        ];
        assert_eq!(
            bits(&grow_from(&[], &far, 40)),
            bits(&grow_by_scan(&[], &far, 40))
        );
        let seed = [[0.0, 0.0, 0.0], [1e15, 0.0, 0.0], [f64::INFINITY, 0.0, 0.0]];
        let unit = Template::FaceCentredCubic.points();
        assert_eq!(
            bits(&grow_from(&seed, &unit, 60)),
            bits(&grow_by_scan(&seed, &unit, 60))
        );
    }

    #[test]
    fn binned_neighbours_are_the_all_pairs_neighbours() {
        let mut r = rng();
        for offsets in shells(&mut r) {
            let sites = grow_from(&[], &offsets, 400);
            let adj = bonded_neighbours(&sites);
            for i in 0..sites.len() {
                let mut got = adj[i].clone();
                got.sort_unstable();
                let want: Vec<usize> = (0..sites.len())
                    .filter(|&j| {
                        let (a, b) = if i < j { (i, j) } else { (j, i) };
                        let d = sq(&sites[a], &sites[b]);
                        j != i && d < 1.44 && d > 1e-9
                    })
                    .collect();
                assert_eq!(got, want, "site {i}");
            }
        }
    }

    #[test]
    fn the_bucketed_peel_removes_what_the_scan_removes() {
        let mut r = rng();
        for offsets in shells(&mut r) {
            let sites = grow_from(&[], &offsets, 300);
            let adj = bonded_neighbours(&sites);
            for n in [1usize, 38, 55, 150, sites.len()] {
                let n = n.min(sites.len());
                for centre in [
                    [0.0, 0.0, 0.0],
                    [
                        r.random::<f64>() - 0.5,
                        r.random::<f64>() - 0.5,
                        r.random::<f64>() - 0.5,
                    ],
                ] {
                    let cd: Vec<f64> = sites.iter().map(|s| sq(s, &centre)).collect();
                    assert_eq!(peel(&adj, &cd, n), peel_by_scan(&sites, centre, n), "n={n}");
                }
            }
        }
    }

    #[test]
    fn a_reused_growth_draws_the_candidate_a_fresh_one_draws() {
        let mut growth = Growth::default();
        let mut x = packed(80);
        let mut first = rng();
        let mut second = rng();
        for step in 0..6 {
            if step % 3 == 2 {
                x[5] += 0.01;
            }
            let a = candidate(Source::Observed, x.view(), 80, &mut first);
            let b = candidate_reusing(&mut growth, Source::Observed, x.view(), 80, &mut second);
            assert_eq!(a.mapv(f64::to_bits), b.mapv(f64::to_bits), "step {step}");
        }
        let y = Array1::from(vec![0.0; 9]);
        let a = candidate(Source::Observed, y.view(), 3, &mut first);
        let b = candidate_reusing(&mut growth, Source::Observed, y.view(), 3, &mut second);
        assert_eq!(a.mapv(f64::to_bits), b.mapv(f64::to_bits));
        assert_eq!(first.random::<u64>(), second.random::<u64>());
    }

    /// An empty or degenerate input must return the input rather than a
    /// malformed proposal.
    #[test]
    fn a_degenerate_input_returns_itself() {
        let mut r = rng();
        let x = Array1::from(vec![0.0, 0.0, 0.0]);
        let c = candidate(Source::Observed, x.view(), 1, &mut r);
        assert_eq!(c.len(), 3);
    }
}
