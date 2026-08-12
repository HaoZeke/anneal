//! Immutable calibration artifact for the fixed census radius.

use super::{DescriptorSignature, SignatureDigest};
use std::collections::BTreeSet;

/// Minimum sample count needed to resolve the 0.99 nearest-rank tail.
pub const MINIMUM_CENSUS_CALIBRATION_PAIRS: usize = 100;

const CENSUS_QUANTILE: f64 = 0.99;

/// Versioned empirical-quantile convention used by a calibration artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalQuantileMethod {
    /// Sorted rank `ceil(q n)`, with ranks numbered from one.
    NearestRankV1,
}

/// Descriptor-distance evidence from two perturbations of one exact minimum.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationPair {
    /// Stable identity of this pair in the archived development pool.
    pub pair_id: String,
    /// Stable identity of the first perturbed configuration.
    pub left_configuration_id: String,
    /// Stable identity of the second perturbed configuration.
    pub right_configuration_id: String,
    /// Exact quenched-minimum identity for the first configuration.
    pub left_minimum_id: String,
    /// Exact quenched-minimum identity for the second configuration.
    pub right_minimum_id: String,
    /// System signature under which both structures were evaluated.
    pub signature_digest: SignatureDigest,
    /// Descriptor distance between the quenched configurations.
    pub distance: f64,
}

/// Input failure that prevents creation of a calibration artifact.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CalibrationError {
    /// The development pool is too small to resolve the requested tail.
    #[error("calibration has {actual} pairs, requires at least {minimum}")]
    InsufficientSamples {
        /// Required number of same-minimum pairs.
        minimum: usize,
        /// Supplied number of pairs.
        actual: usize,
    },
    /// A pair belongs to a different system identity.
    #[error("calibration pair {pair_id} has a foreign system signature")]
    SignatureMismatch {
        /// Stable identity of the rejected pair.
        pair_id: String,
    },
    /// The two configurations did not quench to the same exact minimum.
    #[error("calibration pair {pair_id} does not share an exact minimum")]
    MinimumMismatch {
        /// Stable identity of the rejected pair.
        pair_id: String,
    },
    /// A pair has a NaN or infinite descriptor distance.
    #[error("calibration pair {pair_id} has a nonfinite distance")]
    NonFiniteDistance {
        /// Stable identity of the rejected pair.
        pair_id: String,
    },
    /// A pair has an impossible negative descriptor distance.
    #[error("calibration pair {pair_id} has a negative distance")]
    NegativeDistance {
        /// Stable identity of the rejected pair.
        pair_id: String,
    },
    /// Stable pair identities must be unique within one artifact.
    #[error("calibration pair identity {pair_id} occurs more than once")]
    DuplicatePairId {
        /// Repeated pair identity.
        pair_id: String,
    },
}

/// Fixed census-radius artifact and its complete development evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CensusCalibration {
    signature_digest: SignatureDigest,
    descriptor_schema: DescriptorSignature,
    quantile: f64,
    quantile_method: EmpiricalQuantileMethod,
    pairs: Vec<CalibrationPair>,
    census_radius: f64,
}

impl CensusCalibration {
    /// System identity shared by every raw pair.
    pub fn signature_digest(&self) -> SignatureDigest {
        self.signature_digest
    }

    /// Descriptor schema used to calculate every distance.
    pub fn descriptor_schema(&self) -> &DescriptorSignature {
        &self.descriptor_schema
    }

    /// Requested empirical quantile.
    pub fn quantile(&self) -> f64 {
        self.quantile
    }

    /// Versioned quantile convention.
    pub fn quantile_method(&self) -> EmpiricalQuantileMethod {
        self.quantile_method
    }

    /// Number of raw development pairs.
    pub fn sample_count(&self) -> usize {
        self.pairs.len()
    }

    /// Raw evidence in canonical pair-identity order.
    pub fn pairs(&self) -> &[CalibrationPair] {
        &self.pairs
    }

    /// Calibrated fixed census radius.
    pub fn census_radius(&self) -> f64 {
        self.census_radius
    }
}

/// Build the production 0.99 nearest-rank calibration artifact.
pub fn calibrate_census_radius(
    signature_digest: SignatureDigest,
    descriptor_schema: DescriptorSignature,
    mut pairs: Vec<CalibrationPair>,
) -> Result<CensusCalibration, CalibrationError> {
    if pairs.len() < MINIMUM_CENSUS_CALIBRATION_PAIRS {
        return Err(CalibrationError::InsufficientSamples {
            minimum: MINIMUM_CENSUS_CALIBRATION_PAIRS,
            actual: pairs.len(),
        });
    }

    let mut pair_ids = BTreeSet::new();
    for pair in &pairs {
        if pair.signature_digest != signature_digest {
            return Err(CalibrationError::SignatureMismatch {
                pair_id: pair.pair_id.clone(),
            });
        }
        if pair.left_minimum_id != pair.right_minimum_id {
            return Err(CalibrationError::MinimumMismatch {
                pair_id: pair.pair_id.clone(),
            });
        }
        if !pair.distance.is_finite() {
            return Err(CalibrationError::NonFiniteDistance {
                pair_id: pair.pair_id.clone(),
            });
        }
        if pair.distance < 0.0 {
            return Err(CalibrationError::NegativeDistance {
                pair_id: pair.pair_id.clone(),
            });
        }
        if !pair_ids.insert(pair.pair_id.clone()) {
            return Err(CalibrationError::DuplicatePairId {
                pair_id: pair.pair_id.clone(),
            });
        }
    }

    let mut ranked = pairs.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.pair_id.cmp(&right.pair_id))
    });
    let rank = (CENSUS_QUANTILE * ranked.len() as f64).ceil() as usize;
    let census_radius = ranked[rank - 1].distance;
    pairs.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));

    Ok(CensusCalibration {
        signature_digest,
        descriptor_schema,
        quantile: CENSUS_QUANTILE,
        quantile_method: EmpiricalQuantileMethod::NearestRankV1,
        pairs,
        census_radius,
    })
}
