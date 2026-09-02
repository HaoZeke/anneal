use anneal_core::descriptor_space::{
    DescriptorError, DescriptorGeometry, DescriptorOutputSpec, DescriptorProviderContract,
    DescriptorProviderError, DescriptorProviderInput, DescriptorSpace, InvariantDescriptorProvider,
};
use ndarray::Array1;

#[derive(Debug)]
struct SortedPairDistances {
    contract: DescriptorProviderContract,
    returned_dimension: usize,
}

impl SortedPairDistances {
    fn new(model_digest: [u8; 32], returned_dimension: usize) -> Self {
        Self {
            contract: DescriptorProviderContract::new(
                "test-sorted-pair-distances",
                1,
                model_digest,
                DescriptorOutputSpec::new("feature", 3).unwrap(),
                None,
                3.0,
                "physical-distance-v1",
            )
            .unwrap(),
            returned_dimension,
        }
    }
}

impl InvariantDescriptorProvider for SortedPairDistances {
    fn contract(&self) -> &DescriptorProviderContract {
        &self.contract
    }

    fn describe_system(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Vec<f64>, DescriptorProviderError> {
        let positions = input.coordinates().as_slice().unwrap();
        let atoms = positions.len() / 3;
        let mut distances = Vec::new();
        for left in 0..atoms {
            for right in left + 1..atoms {
                let squared = (0..3)
                    .map(|axis| {
                        let delta = positions[3 * left + axis] - positions[3 * right + axis];
                        delta * delta
                    })
                    .sum::<f64>();
                distances.push(squared.sqrt());
            }
        }
        distances.sort_by(f64::total_cmp);
        distances.truncate(self.returned_dimension);
        Ok(distances)
    }
}

fn descriptor(
    model_digest: [u8; 32],
    returned_dimension: usize,
) -> anneal_core::descriptor_space::DescriptorSpace {
    DescriptorSpace::from_provider(
        DescriptorGeometry::finite(1.0).unwrap(),
        SortedPairDistances::new(model_digest, returned_dimension),
    )
    .unwrap()
}

fn rigid_transform(coordinates: &Array1<f64>) -> Array1<f64> {
    let angle = 0.71_f64;
    let (sine, cosine) = angle.sin_cos();
    let mut transformed = coordinates.clone();
    for atom in 0..coordinates.len() / 3 {
        let x = coordinates[3 * atom];
        let y = coordinates[3 * atom + 1];
        transformed[3 * atom] = cosine * x - sine * y + 1.3;
        transformed[3 * atom + 1] = sine * x + cosine * y - 0.4;
        transformed[3 * atom + 2] = coordinates[3 * atom + 2] + 2.1;
    }
    transformed
}

#[test]
fn provider_values_keep_the_declared_invariant_metric() {
    let coordinates = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.2, 0.1, -0.2, -0.4, 1.1, 0.7]);
    let moved = rigid_transform(&coordinates);
    let permuted = Array1::from_vec(vec![
        coordinates[6],
        coordinates[7],
        coordinates[8],
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
        coordinates[4],
        coordinates[5],
    ]);
    let space = descriptor([7; 32], 3);

    let reference = space
        .describe(coordinates.view(), Some(&[6, 6, 6]))
        .unwrap();
    let transformed = space.describe(moved.view(), Some(&[6, 6, 6])).unwrap();
    let reordered = space.describe(permuted.view(), Some(&[6, 6, 6])).unwrap();

    assert_eq!(reference.values().len(), 3);
    assert!(reference.distance(&transformed).unwrap() < 1e-12);
    assert!(reference.distance(&reordered).unwrap() < 1e-12);
    assert_eq!(space.provider_contract().unwrap().model_digest(), [7; 32]);
}

#[test]
fn provider_digest_is_part_of_descriptor_identity() {
    let coordinates = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.2, 0.1, -0.2, -0.4, 1.1, 0.7]);
    let first = descriptor([3; 32], 3)
        .describe(coordinates.view(), Some(&[6, 6, 6]))
        .unwrap();
    let second = descriptor([4; 32], 3)
        .describe(coordinates.view(), Some(&[6, 6, 6]))
        .unwrap();

    assert_eq!(
        first.distance(&second),
        Err(DescriptorError::IncompatibleDescriptorVectors)
    );
}

#[test]
fn provider_output_shape_is_checked_at_the_boundary() {
    let coordinates = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.2, 0.1, -0.2, -0.4, 1.1, 0.7]);
    let error = descriptor([9; 32], 2)
        .describe(coordinates.view(), Some(&[6, 6, 6]))
        .unwrap_err();

    assert_eq!(
        error,
        DescriptorError::ProviderDimension {
            output: "feature".into(),
            expected: 3,
            actual: 2,
        }
    );
}
