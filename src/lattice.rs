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
        }
    }
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
    let n = x.len() / 3;
    if n < 2 {
        return Vec::new();
    }
    let scale = nearest_neighbour_scale(x);
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

fn sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
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
    let scale = nearest_neighbour_scale(x);
    let offsets = match source {
        Source::Observed => observed_order(x, 1.35),
        Source::Named(t) => t.points(),
    };
    if offsets.is_empty() || n == 0 {
        return x.to_owned();
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

    let mut sites = grow_from(&seed, &offsets, (n * 6).max(n + 32));
    if sites.len() < n {
        return x.to_owned();
    }
    let centre = [
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
    let centre = [
        rng.random::<f64>() - 0.5,
        rng.random::<f64>() - 0.5,
        rng.random::<f64>() - 0.5,
    ];
    let mut live = m;
    while live > n {
        // The least-coordinated survivor, ties going to the one furthest from a
        // jittered centre. The jitter is what makes repeated draws from one
        // source different structures: a body peeled about a site and one
        // peeled about a hole are different clusters, and both are minima of
        // the packing at some size.
        let mut worst = usize::MAX;
        for i in 0..m {
            if !alive[i] {
                continue;
            }
            if worst == usize::MAX
                || count[i] < count[worst]
                || (count[i] == count[worst] && sq(&sites[i], &centre) > sq(&sites[worst], &centre))
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
