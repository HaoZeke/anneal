use anneal_core::catalog::{
    calibrate_census_radius, CalibrationError, CalibrationPair, DescriptorSignature,
    EmpiricalQuantileMethod, SignatureDigest, MINIMUM_CENSUS_CALIBRATION_PAIRS,
};
use std::collections::BTreeMap;

fn descriptor_schema() -> DescriptorSignature {
    DescriptorSignature {
        schema: "multiscale-soap-mean".to_owned(),
        version: 3,
        hyperparameters: BTreeMap::from([
            ("cutoffs".to_owned(), "2.5,3.5,5.0 sigma".to_owned()),
            ("normalization".to_owned(), "block-l2".to_owned()),
        ]),
        species_channels: vec![18],
    }
}

fn pair(index: usize, distance: f64, signature_digest: SignatureDigest) -> CalibrationPair {
    CalibrationPair {
        pair_id: format!("pair-{index:03}"),
        left_configuration_id: format!("perturb-{index:03}-a"),
        right_configuration_id: format!("perturb-{index:03}-b"),
        left_minimum_id: format!("minimum-{:02}", index % 7),
        right_minimum_id: format!("minimum-{:02}", index % 7),
        signature_digest,
        distance,
    }
}

fn calibration_pairs(signature_digest: SignatureDigest) -> Vec<CalibrationPair> {
    (0..MINIMUM_CENSUS_CALIBRATION_PAIRS)
        .map(|index| pair(index, (index + 1) as f64, signature_digest))
        .collect()
}

#[test]
fn census_radius_is_the_nearest_rank_empirical_099_quantile() {
    let signature_digest = [0x41; 32];
    let calibration = calibrate_census_radius(
        signature_digest,
        descriptor_schema(),
        calibration_pairs(signature_digest),
    )
    .unwrap();

    assert_eq!(calibration.signature_digest(), signature_digest);
    assert_eq!(calibration.descriptor_schema(), &descriptor_schema());
    assert_eq!(calibration.quantile(), 0.99);
    assert_eq!(
        calibration.quantile_method(),
        EmpiricalQuantileMethod::NearestRankV1
    );
    assert_eq!(calibration.sample_count(), MINIMUM_CENSUS_CALIBRATION_PAIRS);
    assert_eq!(calibration.census_radius(), 99.0);
    assert_eq!(calibration.pairs().len(), 100);
}

#[test]
fn tied_distances_have_a_canonical_pair_id_order() {
    let signature_digest = [0x42; 32];
    let mut pairs = calibration_pairs(signature_digest);
    pairs[97].distance = 7.0;
    pairs[98].distance = 7.0;
    pairs[99].distance = 7.0;
    pairs.swap(97, 99);

    let calibration =
        calibrate_census_radius(signature_digest, descriptor_schema(), pairs).unwrap();

    assert_eq!(calibration.census_radius(), 96.0);
    let ids = calibration
        .pairs()
        .iter()
        .map(|pair| pair.pair_id.as_str())
        .collect::<Vec<_>>();
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn fewer_than_one_hundred_same_minimum_pairs_are_rejected() {
    let signature_digest = [0x43; 32];
    let mut pairs = calibration_pairs(signature_digest);
    pairs.pop();

    assert_eq!(
        calibrate_census_radius(signature_digest, descriptor_schema(), pairs).unwrap_err(),
        CalibrationError::InsufficientSamples {
            minimum: MINIMUM_CENSUS_CALIBRATION_PAIRS,
            actual: MINIMUM_CENSUS_CALIBRATION_PAIRS - 1,
        }
    );
}

#[test]
fn cross_signature_pairs_are_rejected() {
    let signature_digest = [0x44; 32];
    let mut pairs = calibration_pairs(signature_digest);
    pairs[17].signature_digest = [0x99; 32];

    assert_eq!(
        calibrate_census_radius(signature_digest, descriptor_schema(), pairs).unwrap_err(),
        CalibrationError::SignatureMismatch {
            pair_id: "pair-017".to_owned(),
        }
    );
}

#[test]
fn pairs_that_quench_to_different_minima_are_rejected() {
    let signature_digest = [0x45; 32];
    let mut pairs = calibration_pairs(signature_digest);
    pairs[23].right_minimum_id = "minimum-foreign".to_owned();

    assert_eq!(
        calibrate_census_radius(signature_digest, descriptor_schema(), pairs).unwrap_err(),
        CalibrationError::MinimumMismatch {
            pair_id: "pair-023".to_owned(),
        }
    );
}

#[test]
fn nonfinite_and_negative_distances_are_rejected() {
    let signature_digest = [0x46; 32];
    let mut pairs = calibration_pairs(signature_digest);
    pairs[31].distance = f64::NAN;
    assert_eq!(
        calibrate_census_radius(signature_digest, descriptor_schema(), pairs).unwrap_err(),
        CalibrationError::NonFiniteDistance {
            pair_id: "pair-031".to_owned(),
        }
    );

    let mut pairs = calibration_pairs(signature_digest);
    pairs[32].distance = -0.1;
    assert_eq!(
        calibrate_census_radius(signature_digest, descriptor_schema(), pairs).unwrap_err(),
        CalibrationError::NegativeDistance {
            pair_id: "pair-032".to_owned(),
        }
    );
}
