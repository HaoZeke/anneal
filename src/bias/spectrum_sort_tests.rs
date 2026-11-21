use super::sort_total_order;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn agrees(values: &[f64]) {
    let mut expect = values.to_vec();
    expect.sort_by(f64::total_cmp);
    let mut got = values.to_vec();
    sort_total_order(&mut got);
    let expect: Vec<u64> = expect.iter().map(|v| v.to_bits()).collect();
    let got: Vec<u64> = got.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, expect);
}

#[test]
fn spectra_of_every_length_sort_as_the_comparison_sort_does() {
    let mut rng = StdRng::seed_from_u64(11);
    for n in [0, 1, 2, 63, 64, 65, 100, 1000, 5356, 20_000] {
        let lengths: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 9.0).collect();
        agrees(&lengths);
    }
}

#[test]
fn crowded_equal_and_special_values_sort_as_the_comparison_sort_does() {
    let mut rng = StdRng::seed_from_u64(12);
    agrees(&vec![1.5; 500]);
    let clustered: Vec<f64> = (0..3000)
        .map(|i| {
            if i % 7 == 0 {
                100.0
            } else {
                1.0 + rng.random::<f64>() * 1e-9
            }
        })
        .collect();
    agrees(&clustered);
    let mut special: Vec<f64> = (0..400).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    special.extend([
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
    ]);
    agrees(&special);
}

proptest! {
    #[test]
    fn any_bits_sort_as_the_comparison_sort_does(
        bits in proptest::collection::vec(any::<u64>(), 0..400)
    ) {
        let values: Vec<f64> = bits.into_iter().map(f64::from_bits).collect();
        agrees(&values);
    }
}
