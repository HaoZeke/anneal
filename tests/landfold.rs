use anneal_core::landfold::{
    Landfold, LandfoldConfig, LandfoldMetric, book_descriptors, classical_mds, landfold, parse_book,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Points of a known planar layout, and those points carried isometrically
/// into six dimensions by an orthonormal frame.
fn planar_layout(n: usize, seed: u64) -> (Vec<[f64; 2]>, Vec<Vec<f64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres = [[0.0, 0.0], [3.0, 0.5], [1.0, 2.5], [2.6, 2.8]];
    let plane: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let c = centres[i % centres.len()];
            [
                c[0] + rng.random_range(-0.6..0.6),
                c[1] + rng.random_range(-0.6..0.6),
            ]
        })
        .collect();
    let mut frame = [[0.0; 6]; 2];
    for row in &mut frame {
        for v in row.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
    }
    let dot = |a: &[f64; 6], b: &[f64; 6]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let n0 = dot(&frame[0], &frame[0]).sqrt();
    frame[0].iter_mut().for_each(|v| *v /= n0);
    let proj = dot(&frame[0], &frame[1]);
    let first = frame[0];
    frame[1]
        .iter_mut()
        .zip(first)
        .for_each(|(v, u)| *v -= proj * u);
    let n1 = dot(&frame[1], &frame[1]).sqrt();
    frame[1].iter_mut().for_each(|v| *v /= n1);
    let rows = plane
        .iter()
        .map(|p| {
            (0..6)
                .map(|k| p[0] * frame[0][k] + p[1] * frame[1][k])
                .collect()
        })
        .collect();
    (plane, rows)
}

/// RMS residual of the best rigid motion (rotation, reflection, translation)
/// taking `y` onto `x`, over the RMS spread of `x`.
fn procrustes_residual(x: &[[f64; 2]], y: &[[f64; 2]]) -> f64 {
    fn centred(p: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let n = p.len() as f64;
        let mx = p.iter().map(|v| v[0]).sum::<f64>() / n;
        let my = p.iter().map(|v| v[1]).sum::<f64>() / n;
        p.iter().map(|v| [v[0] - mx, v[1] - my]).collect()
    }
    let x = centred(x);
    let y0 = centred(y);
    let spread = (x.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>() / x.len() as f64).sqrt();
    let mut best = f64::INFINITY;
    for flip in [1.0, -1.0] {
        let y: Vec<[f64; 2]> = y0.iter().map(|v| [flip * v[0], v[1]]).collect();
        let (mut sxx, mut sxy, mut syx, mut syy) = (0.0, 0.0, 0.0, 0.0);
        for (a, b) in x.iter().zip(&y) {
            sxx += b[0] * a[0];
            sxy += b[0] * a[1];
            syx += b[1] * a[0];
            syy += b[1] * a[1];
        }
        let theta = (sxy - syx).atan2(sxx + syy);
        let (s, c) = theta.sin_cos();
        let rms = (x
            .iter()
            .zip(&y)
            .map(|(a, b)| {
                let r = [c * b[0] - s * b[1], s * b[0] + c * b[1]];
                (a[0] - r[0]).powi(2) + (a[1] - r[1]).powi(2)
            })
            .sum::<f64>()
            / x.len() as f64)
            .sqrt();
        best = best.min(rms);
    }
    best / spread
}

fn layout_config() -> LandfoldConfig {
    LandfoldConfig {
        landmarks: 60,
        metric: LandfoldMetric::L2,
        ..LandfoldConfig::default()
    }
}

fn spread(xy: &[[f64; 2]], axis: usize) -> f64 {
    let n = xy.len() as f64;
    let mean = xy.iter().map(|p| p[axis]).sum::<f64>() / n;
    (xy.iter().map(|p| (p[axis] - mean).powi(2)).sum::<f64>() / n).sqrt()
}

fn l1_corners(corners: &[[f64; 2]]) -> Vec<f64> {
    let m = corners.len();
    let mut d = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            d[i * m + j] =
                (corners[i][0] - corners[j][0]).abs() + (corners[i][1] - corners[j][1]).abs();
        }
    }
    d
}

#[test]
fn square_has_lambda1_equal_lambda2_equal_two_and_rectangle_keeps_lambda2() {
    let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let (xy, spectrum) = classical_mds(&l1_corners(&square), 4, 0);
    assert!(
        (spectrum[0] - 2.0).abs() < 1e-9 && (spectrum[1] - 2.0).abs() < 1e-9,
        "square spectrum {spectrum:?}"
    );
    assert!(spectrum[1] > 0.0);
    assert!(spread(&xy, 0) > 0.2 && spread(&xy, 1) > 0.2, "{xy:?}");

    let rectangle = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]];
    let (xy, spectrum) = classical_mds(&l1_corners(&rectangle), 4, 0);
    assert!(spectrum[1] > 0.0, "rectangle lambda2 {spectrum:?}");
    assert!(spectrum[0] > spectrum[1], "{spectrum:?}");
    assert!(spread(&xy, 0) > 0.2 && spread(&xy, 1) > 0.2, "{xy:?}");

    let rows: Vec<Vec<f64>> = square.iter().map(|p| p.to_vec()).collect();
    let fold = landfold(&rows, &LandfoldConfig::default()).unwrap();
    assert!((fold.spectrum[0] - 2.0).abs() < 1e-9 && (fold.spectrum[1] - 2.0).abs() < 1e-9);
    assert!(spread(&fold.xy, 0) > 0.2 && spread(&fold.xy, 1) > 0.2);
}

#[test]
fn known_planar_layout_is_recovered_up_to_rotation() {
    let (plane, rows) = planar_layout(240, 11);
    let fold = landfold(&rows, &layout_config()).unwrap();
    assert_eq!(fold.landmarks.len(), 60);
    let residual = procrustes_residual(&plane, &fold.xy);
    assert!(residual < 1e-4, "relative Procrustes residual {residual}");
    assert!(
        fold.stress < 1e-10,
        "isometric layout has zero stress, got {}",
        fold.stress
    );
    let landmark_plane: Vec<[f64; 2]> = fold.landmarks.iter().map(|&i| plane[i]).collect();
    let landmark_xy: Vec<[f64; 2]> = fold.landmarks.iter().map(|&i| fold.xy[i]).collect();
    assert!(procrustes_residual(&landmark_plane, &landmark_xy) < 1e-6);
}

#[test]
fn a_seed_fixes_the_map_bit_for_bit_at_any_thread_count() {
    let (_, rows) = planar_layout(200, 5);
    let config = LandfoldConfig {
        metric: LandfoldMetric::L1,
        landmarks: 50,
        seed: 7,
        ..LandfoldConfig::default()
    };
    let run = |threads: usize| -> Landfold {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| landfold(&rows, &config).unwrap())
    };
    let one = run(1);
    let four = run(4);
    let again = run(4);
    assert_eq!(one.landmarks, four.landmarks);
    assert_eq!(one.stress.to_bits(), four.stress.to_bits());
    for (a, b) in one
        .xy
        .iter()
        .zip(&four.xy)
        .chain(four.xy.iter().zip(&again.xy))
    {
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert_eq!(a[1].to_bits(), b[1].to_bits());
    }
    let other = landfold(
        &rows,
        &LandfoldConfig {
            seed: 8,
            ..config.clone()
        },
    )
    .unwrap();
    assert_ne!(
        other.landmarks[0], one.landmarks[0],
        "the seed picks the first landmark"
    );
}

#[test]
fn out_of_sample_placement_is_consistent_with_the_fit() {
    let (plane, rows) = planar_layout(300, 23);
    let (fit_rows, held_rows) = rows.split_at(200);
    let fold = landfold(fit_rows, &layout_config()).unwrap();

    let mut is_landmark = vec![false; fit_rows.len()];
    for &i in &fold.landmarks {
        is_landmark[i] = true;
    }
    for (i, row) in fit_rows.iter().enumerate() {
        let placed = fold.map.project(row);
        if is_landmark[i] {
            let drift =
                ((placed[0] - fold.xy[i][0]).powi(2) + (placed[1] - fold.xy[i][1]).powi(2)).sqrt();
            assert!(drift < 1e-6, "landmark {i} moved {drift} when projected");
        } else {
            assert_eq!(placed, fold.xy[i], "row {i}: fit and projection disagree");
        }
    }

    let mut twin = fit_rows.to_vec();
    twin.push(fit_rows[3].clone());
    let with_twin = landfold(&twin, &layout_config()).unwrap();
    assert_eq!(
        with_twin.xy[200], with_twin.xy[3],
        "a duplicate row is placed on its twin"
    );

    let mut held_xy: Vec<[f64; 2]> = held_rows.iter().map(|r| fold.map.project(r)).collect();
    let mut all_xy = fold.xy.clone();
    all_xy.append(&mut held_xy);
    let residual = procrustes_residual(&plane, &all_xy);
    assert!(
        residual < 1e-4,
        "held-out rows off the layout by {residual}"
    );
}

fn quench(pot: &PairPotential, x: Array1<f64>) -> (f64, Vec<f64>) {
    let mut opt = WarmLbfgs::default();
    let (e, x, _) = opt.minimize(x.view(), 4000, |v| Some(pot.value_and_gradient(v)));
    (e, x.to_vec())
}

fn lj38_book() -> String {
    let load = |text: &str| -> Vec<f64> {
        text.lines()
            .skip(2)
            .filter(|l| !l.trim().is_empty())
            .flat_map(|l| {
                l.split_whitespace()
                    .skip(1)
                    .take(3)
                    .map(|t| t.parse::<f64>().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let pot = PairPotential::lennard_jones(38);
    let mut rng = StdRng::seed_from_u64(3);
    let mut book = String::new();
    for fixture in [
        include_str!("fixtures/lj38_fcc.xyz"),
        include_str!("fixtures/lj38_ico.xyz"),
    ] {
        let x0 = load(fixture);
        for k in 0..6 {
            let kick = if k == 0 { 0.0 } else { 0.3 };
            let trial: Vec<f64> = x0
                .iter()
                .map(|v| v + kick * rng.random_range(-1.0..1.0))
                .collect();
            let (e, x) = quench(&pot, Array1::from_vec(trial));
            book.push_str(&format!("{e:.10}"));
            for v in x {
                book.push_str(&format!(" {v:.10}"));
            }
            book.push('\n');
        }
    }
    book
}

#[test]
fn a_decaf_book_folds_with_the_two_lj38_funnels_apart() {
    let book = parse_book(&lj38_book(), false).unwrap();
    let coordinates: Vec<Vec<f64>> = book.iter().map(|m| m.coordinates.clone()).collect();
    let descriptors = book_descriptors(&coordinates).unwrap();
    let width = descriptors.rows[0].len();
    assert!(descriptors.rows.iter().all(|r| r.len() == width));
    let fold = landfold(&descriptors.rows, &LandfoldConfig::default()).unwrap();
    let centre = |range: std::ops::Range<usize>| {
        let n = range.len() as f64;
        let (sx, sy) = range.fold((0.0, 0.0), |(sx, sy), i| {
            (sx + fold.xy[i][0], sy + fold.xy[i][1])
        });
        [sx / n, sy / n]
    };
    let radius = |range: std::ops::Range<usize>, c: [f64; 2]| {
        range
            .map(|i| ((fold.xy[i][0] - c[0]).powi(2) + (fold.xy[i][1] - c[1]).powi(2)).sqrt())
            .fold(0.0, f64::max)
    };
    let fcc = centre(0..6);
    let ico = centre(6..12);
    let gap = ((fcc[0] - ico[0]).powi(2) + (fcc[1] - ico[1]).powi(2)).sqrt();
    assert!(
        gap > radius(0..6, fcc).max(radius(6..12, ico)),
        "fcc and ico overlap on the map: gap {gap}"
    );
}
