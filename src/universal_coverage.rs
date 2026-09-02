//! Uncertainty-aware coverage over the universal stationary-point descriptor.
//!
//! Coverage is evidence for choosing the next exploration target, not a basin
//! identity rule. Exact structural witnesses assign class identifiers. This
//! module combines continuous per-block novelty, Euclidean Gaussian-process
//! uncertainty, disagreement between full and block models, and the graph
//! residual field without applying a descriptor merge radius.

use ndarray::ArrayView1;

use crate::descriptor_space::{DescriptorError, DescriptorVector};
use crate::funnel_bo::FunnelModel;
use crate::residual_field::ResidualField;

/// Gaussian-process scales for the universal exploration acquisition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageConfig {
    /// Full-descriptor GP length scale.
    pub energy_length_scale: f64,
    /// Per-block and optional deep-kernel GP length scale.
    pub component_length_scale: f64,
    /// GP prior standard deviation in energy units.
    pub amplitude: f64,
    /// GP observation noise in energy units.
    pub noise: f64,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            energy_length_scale: 1.0,
            component_length_scale: 0.5,
            amplitude: 10.0,
            noise: 1e-4,
        }
    }
}

impl CoverageConfig {
    fn validate(self) -> Result<Self, CoverageError> {
        for (name, value) in [
            ("energy length scale", self.energy_length_scale),
            ("component length scale", self.component_length_scale),
            ("amplitude", self.amplitude),
            ("noise", self.noise),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(CoverageError::InvalidConfig(name));
            }
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
struct DenseLayer {
    input: usize,
    output: usize,
    weights: Vec<f64>,
    biases: Vec<f64>,
}

/// Deterministic residual neural feature map for a deep-kernel GP.
///
/// The output concatenates the invariant input descriptor with bounded neural
/// features. Consequently its Euclidean distance is never smaller than the raw
/// descriptor distance, so the neural map cannot hide a raw separation. The
/// input is already rotation/permutation invariant, making any deterministic
/// map of it invariant as well. This is a deep-kernel feature map, not a deep
/// Gaussian process: its layer functions are not marginalized as random GPs.
#[derive(Debug, Clone)]
pub struct StableDeepKernel {
    input_dim: usize,
    layers: Vec<DenseLayer>,
}

impl StableDeepKernel {
    /// Construct normalized tanh layers from a reproducible seed.
    pub fn seeded(input_dim: usize, widths: &[usize], seed: u64) -> Result<Self, CoverageError> {
        if input_dim == 0 {
            return Err(CoverageError::InvalidDeepKernel(
                "input dimension must be positive",
            ));
        }
        if widths.is_empty() || widths.contains(&0) {
            return Err(CoverageError::InvalidDeepKernel(
                "every neural layer width must be positive",
            ));
        }
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        let mut layers = Vec::with_capacity(widths.len());
        let mut input = input_dim;
        for &output in widths {
            let scale = 1.0 / (input as f64).sqrt();
            let weights = (0..input * output)
                .map(|_| signed_unit(&mut state) * scale)
                .collect();
            let biases = (0..output).map(|_| 0.1 * signed_unit(&mut state)).collect();
            layers.push(DenseLayer {
                input,
                output,
                weights,
                biases,
            });
            input = output;
        }
        Ok(Self { input_dim, layers })
    }

    /// Required raw descriptor dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Raw skip coordinates followed by the final bounded neural features.
    pub fn embed(&self, input: &[f64]) -> Result<Vec<f64>, CoverageError> {
        if input.len() != self.input_dim {
            return Err(CoverageError::InvalidDeepKernel(
                "input dimension does not match the neural map",
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(CoverageError::InvalidDeepKernel(
                "input contains a nonfinite value",
            ));
        }
        let mut state = input.to_vec();
        for layer in &self.layers {
            let mut next = vec![0.0; layer.output];
            for (row, value) in next.iter_mut().enumerate() {
                let offset = row * layer.input;
                let affine = layer.weights[offset..offset + layer.input]
                    .iter()
                    .zip(&state)
                    .map(|(weight, coordinate)| weight * coordinate)
                    .sum::<f64>()
                    + layer.biases[row];
                *value = affine.tanh();
            }
            state = next;
        }
        let scale = 1.0 / (state.len() as f64).sqrt();
        let mut embedded = Vec::with_capacity(input.len() + state.len());
        embedded.extend_from_slice(input);
        embedded.extend(state.into_iter().map(|value| scale * value));
        Ok(embedded)
    }

    /// Stable Euclidean distance in the residual neural feature space.
    pub fn distance(&self, left: &[f64], right: &[f64]) -> Result<f64, CoverageError> {
        let left = self.embed(left)?;
        let right = self.embed(right)?;
        Ok(euclidean(&left, &right))
    }
}

/// Evidence used to rank one stationary-structure exploration target.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageEvidence {
    /// Nearest observed distance for every ordered descriptor block.
    pub nearest_block_distances: Vec<Option<f64>>,
    /// RMS of smoothly squashed nearest-block distances.
    pub block_novelty: f64,
    /// Full-descriptor GP posterior mean energy.
    pub energy_mean: f64,
    /// Full-descriptor GP posterior standard deviation.
    pub energy_standard_deviation: f64,
    /// Per-block GP posterior mean energies.
    pub block_means: Vec<f64>,
    /// Per-block GP posterior standard deviations.
    pub block_standard_deviations: Vec<f64>,
    /// Optional residual-neural deep-kernel GP posterior mean.
    pub deep_kernel_mean: Option<f64>,
    /// Optional residual-neural deep-kernel GP posterior standard deviation.
    pub deep_kernel_standard_deviation: Option<f64>,
    /// Uniform model-average posterior mean across invariant descriptor views.
    pub ensemble_mean: f64,
    /// Moment-matched posterior standard deviation, including disagreement.
    pub ensemble_standard_deviation: f64,
    /// Standard deviation of the posterior means, scaled by the GP amplitude.
    pub model_disagreement: f64,
    /// Effort-adjusted graph-GMRF variance, or the unassigned residual variance.
    pub residual_variance: f64,
    /// Moment-matched model-average expected improvement for minimization.
    pub expected_improvement: f64,
    /// Expected improvement divided by the declared system energy scale.
    pub acquisition: f64,
}

#[derive(Debug, Clone, Copy)]
struct BlockLayout {
    offset: usize,
    len: usize,
}

/// GP/GMRF evidence model over exact stationary-point classes.
#[derive(Debug, Clone)]
pub struct UniversalCoverage {
    template: DescriptorVector,
    config: CoverageConfig,
    blocks: Vec<BlockLayout>,
    class_representatives: Vec<Option<Vec<f64>>>,
    observed_classes: Vec<bool>,
    observation_count: usize,
    energy: FunnelModel,
    block_models: Vec<FunnelModel>,
    deep_kernel: Option<StableDeepKernel>,
    deep_model: Option<FunnelModel>,
    residual: ResidualField,
}

impl UniversalCoverage {
    /// Construct full-descriptor, per-block, and graph residual models.
    pub fn new(
        reference: &DescriptorVector,
        config: CoverageConfig,
    ) -> Result<Self, CoverageError> {
        Self::build(reference, config, None)
    }

    /// Construct coverage with an additional residual-neural deep-kernel GP.
    pub fn with_deep_kernel(
        reference: &DescriptorVector,
        config: CoverageConfig,
        deep_kernel: StableDeepKernel,
    ) -> Result<Self, CoverageError> {
        Self::build(reference, config, Some(deep_kernel))
    }

    fn build(
        reference: &DescriptorVector,
        config: CoverageConfig,
        deep_kernel: Option<StableDeepKernel>,
    ) -> Result<Self, CoverageError> {
        let config = config.validate()?;
        if reference.values().is_empty() || reference.blocks().is_empty() {
            return Err(CoverageError::InvalidConfig(
                "reference descriptor must contain blocks",
            ));
        }
        if deep_kernel
            .as_ref()
            .is_some_and(|kernel| kernel.input_dim() != reference.values().len())
        {
            return Err(CoverageError::InvalidDeepKernel(
                "neural input dimension does not match the descriptor",
            ));
        }
        let blocks = reference
            .blocks()
            .iter()
            .map(|block| BlockLayout {
                offset: block.offset(),
                len: block.len(),
            })
            .collect::<Vec<_>>();
        let block_models = blocks
            .iter()
            .map(|_| {
                FunnelModel::new_euclidean(
                    config.component_length_scale,
                    config.amplitude,
                    config.noise,
                )
            })
            .collect();
        let deep_model = deep_kernel.as_ref().map(|_| {
            FunnelModel::new_euclidean(
                config.component_length_scale,
                config.amplitude,
                config.noise,
            )
        });
        Ok(Self {
            template: reference.clone(),
            config,
            blocks,
            class_representatives: Vec::new(),
            observed_classes: Vec::new(),
            observation_count: 0,
            energy: FunnelModel::new_euclidean(
                config.energy_length_scale,
                config.amplitude,
                config.noise,
            ),
            block_models,
            deep_kernel,
            deep_model,
            residual: ResidualField::new(),
        })
    }

    /// Record an energy observation assigned by an exact structural witness.
    pub fn observe(
        &mut self,
        class: usize,
        descriptor: &DescriptorVector,
        energy: f64,
    ) -> Result<(), CoverageError> {
        self.ensure_compatible(descriptor)?;
        self.observe_values(class, descriptor.values(), energy)
    }

    /// Record schema-bound descriptor values assigned by an exact witness.
    pub fn observe_values(
        &mut self,
        class: usize,
        descriptor: &[f64],
        energy: f64,
    ) -> Result<(), CoverageError> {
        self.ensure_values(descriptor)?;
        if !energy.is_finite() {
            return Err(CoverageError::NonFiniteEnergy);
        }
        if class >= self.observed_classes.len() {
            self.observed_classes.resize(class + 1, false);
            self.class_representatives.resize(class + 1, None);
        }
        if !self.observed_classes[class] {
            self.observed_classes[class] = true;
            self.class_representatives[class] = Some(descriptor.to_vec());
        }
        let representative = self.class_representatives[class]
            .as_deref()
            .expect("an observed exact class has a representative");
        self.residual.observe(class, energy);
        self.energy
            .observe(ArrayView1::from(representative), energy);
        for (model, layout) in self.block_models.iter_mut().zip(&self.blocks) {
            model.observe(
                ArrayView1::from(block_values(representative, *layout)),
                energy,
            );
        }
        if let (Some(kernel), Some(model)) = (&self.deep_kernel, &mut self.deep_model) {
            let embedded = kernel.embed(representative)?;
            model.observe(ArrayView1::from(embedded.as_slice()), energy);
        }
        self.observation_count = self.observation_count.saturating_add(1);
        Ok(())
    }

    /// Add an observed transition edge between exact classes.
    pub fn connect(&mut self, left: usize, right: usize) -> Result<(), CoverageError> {
        self.ensure_class(left)?;
        self.ensure_class(right)?;
        self.residual.edge(left, right);
        Ok(())
    }

    /// Number of class identifiers admitted by exact witnesses.
    pub fn exact_class_count(&self) -> usize {
        self.observed_classes.iter().filter(|&&seen| seen).count()
    }

    /// Number of energy/descriptor observations, including repeat effort.
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Assemble novelty, posterior uncertainty, disagreement, and graph evidence.
    pub fn evidence(
        &mut self,
        descriptor: &DescriptorVector,
        assigned_class: Option<usize>,
    ) -> Result<CoverageEvidence, CoverageError> {
        self.ensure_compatible(descriptor)?;
        self.evidence_values(descriptor.values(), assigned_class)
    }

    /// Assemble evidence from values already validated against this schema.
    pub fn evidence_values(
        &mut self,
        descriptor: &[f64],
        assigned_class: Option<usize>,
    ) -> Result<CoverageEvidence, CoverageError> {
        self.ensure_values(descriptor)?;
        if let Some(class) = assigned_class {
            self.ensure_class(class)?;
        }
        let values = ArrayView1::from(descriptor);
        let (energy_mean, energy_standard_deviation) = self.energy.predict(values);
        let mut block_means = Vec::with_capacity(self.blocks.len());
        let mut block_standard_deviations = Vec::with_capacity(self.blocks.len());
        for (model, layout) in self.block_models.iter_mut().zip(&self.blocks) {
            let (mean, standard_deviation) =
                model.predict(ArrayView1::from(block_values(descriptor, *layout)));
            block_means.push(mean);
            block_standard_deviations.push(standard_deviation);
        }
        let (deep_kernel_mean, deep_kernel_standard_deviation) =
            if let (Some(kernel), Some(model)) = (&self.deep_kernel, &mut self.deep_model) {
                let embedded = kernel.embed(descriptor)?;
                let (mean, standard_deviation) =
                    model.predict(ArrayView1::from(embedded.as_slice()));
                (Some(mean), Some(standard_deviation))
            } else {
                (None, None)
            };
        let nearest_block_distances = self.nearest_block_distances(descriptor);
        let block_novelty = novelty(&nearest_block_distances);
        let mut means =
            Vec::with_capacity(1 + block_means.len() + usize::from(deep_kernel_mean.is_some()));
        means.push(energy_mean);
        means.extend(block_means.iter().copied());
        means.extend(deep_kernel_mean);
        let model_disagreement = standard_deviation(&means) / self.config.amplitude;
        let mut standard_deviations = Vec::with_capacity(
            1 + block_standard_deviations.len()
                + usize::from(deep_kernel_standard_deviation.is_some()),
        );
        standard_deviations.push(energy_standard_deviation);
        standard_deviations.extend(block_standard_deviations.iter().copied());
        standard_deviations.extend(deep_kernel_standard_deviation);
        let ensemble_mean = means.iter().sum::<f64>() / means.len() as f64;
        let ensemble_variance = means
            .iter()
            .zip(&standard_deviations)
            .map(|(mean, standard_deviation)| {
                standard_deviation * standard_deviation
                    + (mean - ensemble_mean) * (mean - ensemble_mean)
            })
            .sum::<f64>()
            / means.len() as f64;
        let ensemble_standard_deviation = ensemble_variance.max(0.0).sqrt();
        let residual_variance = assigned_class
            .map(|class| self.residual.score(class))
            .unwrap_or_else(|| self.residual.residual_score());
        let expected_improvement = gaussian_expected_improvement(
            self.energy.incumbent(),
            ensemble_mean,
            ensemble_standard_deviation,
            self.config.amplitude,
        );
        let acquisition = expected_improvement / self.config.amplitude;
        Ok(CoverageEvidence {
            nearest_block_distances,
            block_novelty,
            energy_mean,
            energy_standard_deviation,
            block_means,
            block_standard_deviations,
            deep_kernel_mean,
            deep_kernel_standard_deviation,
            ensemble_mean,
            ensemble_standard_deviation,
            model_disagreement,
            residual_variance,
            expected_improvement,
            acquisition,
        })
    }

    fn nearest_block_distances(&self, descriptor: &[f64]) -> Vec<Option<f64>> {
        self.blocks
            .iter()
            .map(|&layout| {
                self.class_representatives
                    .iter()
                    .flatten()
                    .map(|observed| {
                        euclidean(
                            block_values(descriptor, layout),
                            block_values(observed, layout),
                        )
                    })
                    .min_by(f64::total_cmp)
            })
            .collect()
    }

    fn ensure_compatible(&self, descriptor: &DescriptorVector) -> Result<(), CoverageError> {
        self.template.distance(descriptor)?;
        Ok(())
    }

    fn ensure_values(&self, descriptor: &[f64]) -> Result<(), CoverageError> {
        if descriptor.len() != self.template.values().len() {
            return Err(CoverageError::DescriptorDimension {
                expected: self.template.values().len(),
                actual: descriptor.len(),
            });
        }
        if let Some(index) = descriptor.iter().position(|value| !value.is_finite()) {
            return Err(CoverageError::NonFiniteDescriptor { index });
        }
        Ok(())
    }

    fn ensure_class(&self, class: usize) -> Result<(), CoverageError> {
        if self.observed_classes.get(class).copied().unwrap_or(false) {
            Ok(())
        } else {
            Err(CoverageError::UnknownClass { class })
        }
    }
}

/// Invalid coverage model input or evidence request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CoverageError {
    /// A configuration value violates its numerical domain.
    #[error("invalid universal coverage configuration: {0}")]
    InvalidConfig(&'static str),
    /// A neural feature-map shape or value is invalid.
    #[error("invalid stable deep kernel: {0}")]
    InvalidDeepKernel(&'static str),
    /// Only finite stationary energies can train the evidence models.
    #[error("universal coverage energy must be finite")]
    NonFiniteEnergy,
    /// Schema-bound values have the wrong descriptor dimension.
    #[error("descriptor dimension is {actual}, expected {expected}")]
    DescriptorDimension {
        /// Descriptor dimension fixed at model construction.
        expected: usize,
        /// Descriptor dimension supplied by the caller.
        actual: usize,
    },
    /// Schema-bound values contain NaN or infinity.
    #[error("nonfinite descriptor value at index {index}")]
    NonFiniteDescriptor {
        /// Index of the first invalid coordinate.
        index: usize,
    },
    /// A graph edge or assigned query names a class without observations.
    #[error("unknown exact stationary class {class}")]
    UnknownClass {
        /// Unobserved class identifier.
        class: usize,
    },
    /// Descriptor schema or value compatibility failure.
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
}

fn block_values(descriptor: &[f64], layout: BlockLayout) -> &[f64] {
    &descriptor[layout.offset..layout.offset + layout.len]
}

fn novelty(distances: &[Option<f64>]) -> f64 {
    if distances.iter().all(Option::is_none) {
        return 1.0;
    }
    let values = distances
        .iter()
        .map(|distance| distance.map_or(1.0, |value| value / (1.0 + value)))
        .collect::<Vec<_>>();
    root_mean_square(&values)
}

fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| {
            let difference = value - mean;
            difference * difference
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn root_mean_square(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn euclidean(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f64>()
        .sqrt()
}

fn gaussian_expected_improvement(
    incumbent: Option<f64>,
    mean: f64,
    standard_deviation: f64,
    prior_scale: f64,
) -> f64 {
    let Some(incumbent) = incumbent else {
        return prior_scale;
    };
    if standard_deviation <= 1e-12 {
        return (incumbent - mean).max(0.0);
    }
    let z = (incumbent - mean) / standard_deviation;
    ((incumbent - mean) * normal_cdf(z) + standard_deviation * normal_pdf(z)).max(0.0)
}

fn normal_pdf(value: f64) -> f64 {
    (-0.5 * value * value).exp() / std::f64::consts::TAU.sqrt()
}

fn normal_cdf(value: f64) -> f64 {
    0.5 * (1.0 + erf(value / std::f64::consts::SQRT_2))
}

fn erf(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let value = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * value);
    let approximation = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-value * value).exp();
    sign * approximation
}

fn signed_unit(state: &mut u64) -> f64 {
    let raw = splitmix64(state) as f64 / u64::MAX as f64;
    2.0 * raw - 1.0
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
