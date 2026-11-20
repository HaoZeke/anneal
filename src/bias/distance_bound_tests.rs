use super::{BasinIndex, BasinMetric, BlockSums, EuclideanMetric, Fingerprint};
use ndarray::{Array1, ArrayView1};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The decision `distance_bounded` is allowed to shortcut: the
/// front-to-back sum, infinite once it is past `bound * bound`.
fn block_sum(a: &[f64], b: &[f64], bound: f64) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    if acc > bound * bound {
        f64::INFINITY
    } else {
        acc.sqrt()
    }
}

fn squared(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

fn bounded(a: &[f64], b: &[f64], bound: f64) -> f64 {
    EuclideanMetric.distance_bounded(ArrayView1::from(a), ArrayView1::from(b), bound)
}

fn assert_same_bits(a: &[f64], b: &[f64], bound: f64) {
    let got = bounded(a, b, bound);
    let want = block_sum(a, b, bound);
    assert!(
        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
        "len {} bound {bound:e}: bounded {got:e}, block sum {want:e}",
        a.len()
    );
}

/// A sorted-spectrum pair: `a` ascending, `b` a perturbation of it.
fn spectrum(rng: &mut StdRng, n: usize, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let mut a: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * scale).collect();
    a.sort_by(f64::total_cmp);
    let b = a
        .iter()
        .map(|x| x + (rng.random::<f64>() - 0.5) * scale)
        .collect();
    (a, b)
}

#[test]
fn a_centre_on_the_radius_is_kept_at_any_descriptor_scale() {
    // The squared distance is the radius squared to the last bit. Every
    // early stop must fall through to the block sum, which keeps it.
    let mut rng = StdRng::seed_from_u64(0x5eed_b1a5);
    for scale in [1.0e-3, 1.0, 1.0e2, 1.0e4, 1.0e5, 1.0e6, 1.0e8] {
        for n in [33, 64, 80, 100, 256, 300, 1485] {
            for _ in 0..16 {
                let (a, b) = spectrum(&mut rng, n, scale);
                let bound = squared(&a, &b).sqrt();
                let at = block_sum(&a, &b, bound);
                if at.is_finite() {
                    assert_same_bits(&a, &b, bound);
                }
                assert_same_bits(&a, &b, bound * (1.0 + 4.0 * f64::EPSILON));
            }
        }
    }
}

#[test]
fn short_and_mismatched_descriptors_keep_the_plain_answer() {
    for n in 0..20 {
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 + 0.25).collect();
        for bound in [0.0, 0.1, 1.0, 10.0, f64::INFINITY] {
            assert_same_bits(&a, &b, bound);
        }
    }
    assert_eq!(bounded(&[0.0; 40], &[0.0; 39], 1.0), f64::INFINITY);
}

#[test]
fn a_non_finite_descriptor_is_never_inside_the_radius() {
    let mut rng = StdRng::seed_from_u64(7);
    for n in [1, 17, 40, 64, 130, 300] {
        let (a, mut b) = spectrum(&mut rng, n, 1.0);
        for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for at in [0, n / 2, n - 1] {
                let held = b[at];
                b[at] = poison;
                for bound in [1.0e-3, 1.0, 1.0e3, f64::MAX] {
                    let got = bounded(&a, &b, bound);
                    assert!(!(got <= bound), "n {n} poison at {at}: {got}");
                }
                b[at] = held;
            }
        }
    }
}

fn magnitude() -> impl Strategy<Value = f64> {
    (-3i32..=9).prop_map(|exponent| 10f64.powi(exponent))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn bounded_distance_is_the_block_sum(
        seed in any::<u64>(),
        n in 0usize..400,
        scale in magnitude(),
        nudge in -8i32..=8,
        free in 0.0f64..2.0,
        on_edge in any::<bool>(),
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (a, b) = spectrum(&mut rng, n, scale);
        let exact = squared(&a, &b).sqrt();
        let bound = if on_edge {
            exact * (1.0 + f64::from(nudge) * f64::EPSILON)
        } else {
            exact * free
        };
        let got = bounded(&a, &b, bound);
        let want = block_sum(&a, &b, bound);
        prop_assert_eq!(got.to_bits(), want.to_bits(), "n {} bound {:e}", n, bound);
    }
}

#[derive(Clone, Copy)]
struct Raw;

impl Fingerprint for Raw {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// `lookup` is the newest centre the block sum places inside the radius.
    #[test]
    fn lookup_is_the_newest_centre_inside_the_radius(
        seed in any::<u64>(),
        n in 1usize..300,
        centres in 1usize..12,
        scale in magnitude(),
        radius_fraction in 0.05f64..1.5,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (query, _) = spectrum(&mut rng, n, scale);
        let radius = radius_fraction * scale;
        let mut index = BasinIndex::new(Raw, radius);
        let mut held = Vec::new();
        for _ in 0..centres {
            let spread = scale * rng.random::<f64>() * 2.0 / (n as f64).sqrt();
            let centre: Vec<f64> = query
                .iter()
                .map(|x| x + (rng.random::<f64>() - 0.5) * spread)
                .collect();
            index.push(Array1::from(centre.clone()));
            held.push(centre);
        }
        let brute = held
            .iter()
            .enumerate()
            .rev()
            .find(|(_, centre)| block_sum(centre, &query, radius) <= radius)
            .map(|(i, _)| i);
        prop_assert_eq!(index.lookup(ArrayView1::from(&query[..])), brute);
    }
}

/// The lookup as it was before the block bound: the newest centre that
/// passes the key, the suffix and the bounded distance.
fn old_walk<F: Fingerprint>(index: &BasinIndex<F>, d: ArrayView1<f64>) -> Option<usize> {
    let dk = index.metric.key(d);
    for (i, c) in index.centres.iter().enumerate().rev() {
        if c.len() != d.len() {
            continue;
        }
        if let (Some(k), Some(ck)) = (dk, index.keys[i])
            && (k - ck).abs() > index.merge_radius
        {
            continue;
        }
        if index.suffix_separates(i, d) {
            continue;
        }
        if index
            .metric
            .distance_bounded(c.view(), d, index.merge_radius)
            <= index.merge_radius
        {
            return Some(i);
        }
    }
    None
}

/// A centre at `fraction` of the radius from `query` along a random
/// direction, so the stored set straddles the radius.
fn at_fraction(rng: &mut StdRng, query: &[f64], radius: f64, fraction: f64) -> Vec<f64> {
    let direction: Vec<f64> = query.iter().map(|_| rng.random::<f64>() - 0.5).collect();
    let norm = direction
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    query
        .iter()
        .zip(&direction)
        .map(|(q, v)| q + v / norm * radius * fraction)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Every lookup and every registration matches the old walk, on centres
    /// spread from inside the radius to several radii out.
    #[test]
    fn the_block_bound_keeps_every_answer_of_the_old_walk(
        seed in any::<u64>(),
        n in 1usize..400,
        centres in 1usize..80,
        scale in magnitude(),
        radius_fraction in 0.01f64..2.0,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (query, _) = spectrum(&mut rng, n, scale);
        let radius = radius_fraction * scale;
        let mut index = BasinIndex::new(Raw, radius);
        let mut probes = vec![query.clone()];
        for _ in 0..centres {
            let fraction = rng.random::<f64>() * 4.0;
            let centre = at_fraction(&mut rng, &query, radius, fraction);
            prop_assert_eq!(
                index.lookup(ArrayView1::from(&centre[..])),
                old_walk(&index, ArrayView1::from(&centre[..]))
            );
            index.push(Array1::from(centre.clone()));
            let reach = rng.random::<f64>() * 2.0;
            probes.push(at_fraction(&mut rng, &centre, radius, reach));
        }
        for probe in &probes {
            let view = ArrayView1::from(&probe[..]);
            prop_assert_eq!(index.lookup(view), old_walk(&index, view), "n {} radius {:e}", n, radius);
        }
    }

    /// A centre the block bound skips is one the bounded distance puts
    /// outside the radius, including centres within a few ulps of it.
    #[test]
    fn a_skipped_centre_is_outside_the_radius(
        seed in any::<u64>(),
        n in 1usize..2000,
        scale in magnitude(),
        nudge in -8i32..=8,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (a, b) = spectrum(&mut rng, n, scale);
        let exact = squared(&a, &b).sqrt();
        let radius = exact * (1.0 + f64::from(nudge) * f64::EPSILON);
        if radius > 0.0 {
            let mut index = BasinIndex::new(Raw, radius);
            index.push(Array1::from(a.clone()));
            let query = BlockSums::of(ArrayView1::from(&b[..]));
            if index.blocks_separate(0, &query) {
                prop_assert!(!(bounded(&a, &b, radius) <= radius));
            }
        }
    }
}

#[test]
fn the_block_bound_skips_far_centres_and_keeps_the_rest() {
    // Not vacuous: on centres one to four radii out the bound dismisses
    // most of what lies past two radii and nothing the distance keeps.
    let mut rng = StdRng::seed_from_u64(0xb10c_5eed);
    let (query, _) = spectrum(&mut rng, 3160, 5.0);
    let radius = 0.7;
    let mut index = BasinIndex::new(Raw, radius);
    let probe = BlockSums::of(ArrayView1::from(&query[..]));
    let (mut far, mut skipped) = (0usize, 0usize);
    for k in 0..400 {
        let fraction = 1.0 + 3.0 * k as f64 / 400.0;
        let mut centre = at_fraction(&mut rng, &query, radius, fraction);
        // A uniform shift of the long end is what separates two isomers'
        // spectra, and what a block sum sees.
        let len = centre.len();
        for v in &mut centre[len - 256..] {
            *v += radius * fraction / 16.0;
        }
        index.push(Array1::from(centre.clone()));
        let i = index.n_basins() - 1;
        let separated = index.blocks_separate(i, &probe);
        let inside = bounded(&centre, &query, radius) <= radius;
        assert!(
            !(separated && inside),
            "centre {i} skipped inside the radius"
        );
        if fraction >= 2.0 {
            far += 1;
            skipped += usize::from(separated);
        }
    }
    assert!(
        skipped * 2 > far,
        "skipped {skipped} of {far} centres past two radii"
    );
    assert_eq!(
        index.lookup(ArrayView1::from(&query[..])),
        old_walk(&index, ArrayView1::from(&query[..]))
    );
}

#[test]
fn a_non_finite_centre_or_query_turns_the_block_bound_off() {
    let mut rng = StdRng::seed_from_u64(11);
    let (a, b) = spectrum(&mut rng, 100, 1.0);
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = a.clone();
        bad[50] = poison;
        let mut index = BasinIndex::new(Raw, 1.0e-3);
        index.push(Array1::from(bad.clone()));
        index.push(Array1::from(a.clone()));
        let clean = BlockSums::of(ArrayView1::from(&b[..]));
        let poisoned = BlockSums::of(ArrayView1::from(&bad[..]));
        assert!(!index.blocks_separate(0, &clean));
        assert!(!index.blocks_separate(1, &poisoned));
        for q in [&a, &b, &bad] {
            let view = ArrayView1::from(&q[..]);
            assert_eq!(index.lookup(view), old_walk(&index, view));
        }
    }
}

#[test]
fn a_metric_that_is_not_euclidean_never_uses_the_block_bound() {
    struct Coarse;
    impl BasinMetric for Coarse {
        fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
            // Only the first entry counts, so the block sums of the rest
            // say nothing about it.
            (a[0] - b[0]).abs()
        }
    }
    let mut index = BasinIndex::new(Raw, 0.5).with_metric(Box::new(Coarse));
    // At 32 entries the suffix test is off, so only the block bound could
    // skip this centre.
    let centre: Vec<f64> = (0..32).map(|k| k as f64).collect();
    let mut query = centre.clone();
    for v in &mut query[1..] {
        *v += 100.0;
    }
    index.push(Array1::from(centre));
    assert_eq!(index.lookup(ArrayView1::from(&query[..])), Some(0));
}
