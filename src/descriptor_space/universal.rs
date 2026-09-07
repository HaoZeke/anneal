//! Fixed-dimensional invariant geometry for cross-system basin catalogues.

use super::{
    CONTRACTIVE_L2_NORMALIZATION_SCHEMA, DescriptorBlockKind, DescriptorBlockMetadata,
    DescriptorBlockSpec, DescriptorError, DescriptorSchema, DescriptorSpace, DescriptorVector,
};
use crate::soap::central_spectrum_from_displacements;
use linkcell::Cell;
use ndarray::{Array2, ArrayView1};
use std::f64::consts::{PI, TAU};

/// Stable schema name for the cross-system descriptor.
pub const UNIVERSAL_DESCRIPTOR_SCHEMA: &str = "anneal-universal-pes";
/// Stable schema version for the cross-system descriptor.
pub const UNIVERSAL_DESCRIPTOR_VERSION: u32 = 2;
/// Euclidean radius for block-balanced universal local environments.
pub const UNIVERSAL_LOCAL_ENVIRONMENT_RADIUS: f64 = 0.2;

const PAIR_CHANNELS: usize = 4;
const TRIPLE_CHANNELS: usize = 3;
const SPECIES_CHANNELS: usize = 3;
const GRAPH_FEATURES: usize = 10;

/// Length scale, simulation cell, and periodic axes used by one descriptor space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescriptorGeometry {
    length_scale: f64,
    cell: Option<[f64; 9]>,
    periodic: [bool; 3],
}

impl DescriptorGeometry {
    /// Construct nonperiodic geometry in units of `length_scale`.
    pub fn finite(length_scale: f64) -> Result<Self, DescriptorError> {
        Self::new(length_scale, None, [false; 3])
    }

    /// Construct geometry with row-major cell vectors and selected periodic axes.
    pub fn new(
        length_scale: f64,
        cell: Option<[f64; 9]>,
        periodic: [bool; 3],
    ) -> Result<Self, DescriptorError> {
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(DescriptorError::InvalidLengthScale);
        }
        if periodic.iter().any(|&axis| axis) && cell.is_none() {
            return Err(DescriptorError::PeriodicCellRequired);
        }
        if let Some(cell) = cell
            && (cell.iter().any(|value| !value.is_finite()) || determinant(cell).abs() <= 1e-12)
        {
            return Err(DescriptorError::InvalidCell);
        }
        Ok(Self {
            length_scale,
            cell,
            periodic,
        })
    }

    /// Coordinate unit represented by one descriptor length.
    pub fn length_scale(self) -> f64 {
        self.length_scale
    }

    /// Row-major cell vectors, when supplied.
    pub fn cell(self) -> Option<[f64; 9]> {
        self.cell
    }

    /// Periodic lattice-vector flags.
    pub fn periodic(self) -> [bool; 3] {
        self.periodic
    }

    pub(crate) fn displacement(self, delta: [f64; 3]) -> [f64; 3] {
        let mut displacement = delta;
        if let Some(cell) = self.cell
            && self.periodic.iter().any(|&axis| axis)
        {
            let a = [cell[0], cell[1], cell[2]];
            let b = [cell[3], cell[4], cell[5]];
            let c = [cell[6], cell[7], cell[8]];
            let det = dot(a, cross(b, c));
            let mut fractional = [
                dot(delta, cross(b, c)) / det,
                dot(delta, cross(c, a)) / det,
                dot(delta, cross(a, b)) / det,
            ];
            for axis in 0..3 {
                if self.periodic[axis] {
                    fractional[axis] -= fractional[axis].round();
                }
            }
            displacement = [
                fractional[0] * a[0] + fractional[1] * b[0] + fractional[2] * c[0],
                fractional[0] * a[1] + fractional[1] * b[1] + fractional[2] * c[1],
                fractional[0] * a[2] + fractional[1] * b[2] + fractional[2] * c[2],
            ];
        }
        [
            displacement[0] / self.length_scale,
            displacement[1] / self.length_scale,
            displacement[2] / self.length_scale,
        ]
    }
}

/// Construct the immutable descriptor shared by clusters, molecules, and surfaces.
pub fn universal_descriptor_space(geometry: DescriptorGeometry) -> DescriptorSpace {
    let blocks = vec![
        block(DescriptorBlockKind::PairRadial, 12, 3, 2.5),
        block(DescriptorBlockKind::PairRadial, 16, 3, 6.0),
        block(DescriptorBlockKind::ThreeBodyAngular, 3, 4, 3.0),
        block(DescriptorBlockKind::ThreeBodyAngular, 3, 5, 6.0),
        block(DescriptorBlockKind::GraphTopology, 6, 0, 6.0),
        block(DescriptorBlockKind::InvariantSoapMean, 3, 4, 3.0),
        block(DescriptorBlockKind::InvariantSoapMean, 3, 6, 6.0),
        block(DescriptorBlockKind::InvariantAceNu3Mean, 2, 3, 3.0),
        block(DescriptorBlockKind::InvariantAceNu3Mean, 2, 4, 6.0),
        block(DescriptorBlockKind::ChiralMoment, 3, 0, 3.0),
        block(DescriptorBlockKind::ChiralMoment, 3, 0, 6.0),
    ];
    let schema = DescriptorSchema::new(
        UNIVERSAL_DESCRIPTOR_SCHEMA,
        UNIVERSAL_DESCRIPTOR_VERSION,
        blocks,
    )
    .expect("the built-in universal descriptor schema is valid");
    DescriptorSpace::with_geometry(schema, geometry)
}

fn block(
    kind: DescriptorBlockKind,
    n_max: usize,
    l_max: usize,
    cutoff: f64,
) -> DescriptorBlockSpec {
    DescriptorBlockSpec::new(kind, n_max, l_max, cutoff)
        .expect("the built-in universal descriptor block is valid")
}

pub(super) fn describe(
    schema: &DescriptorSchema,
    geometry: DescriptorGeometry,
    coordinates: ArrayView1<f64>,
    species: Option<&[u32]>,
) -> Result<DescriptorVector, DescriptorError> {
    let maximum_cutoff = schema
        .blocks
        .iter()
        .map(|block| block.cutoff())
        .fold(0.0, f64::max);
    let environment = Environment::new(geometry, coordinates, species, maximum_cutoff)?;
    let mut values = Vec::new();
    let mut metadata = Vec::with_capacity(schema.blocks.len());
    for (block_index, block) in schema.blocks.iter().copied().enumerate() {
        let mut aggregated = match block.kind() {
            DescriptorBlockKind::PairRadial => pair_radial(&environment, block),
            DescriptorBlockKind::ThreeBodyAngular => three_body_angular(&environment, block),
            DescriptorBlockKind::GraphTopology => graph_topology(&environment, block),
            DescriptorBlockKind::InvariantSoapMean => {
                invariant_spectrum(&environment, block, false)
            }
            DescriptorBlockKind::InvariantAceNu3Mean => {
                invariant_spectrum(&environment, block, true)
            }
            DescriptorBlockKind::ChiralMoment => chiral_moments(&environment, block),
            DescriptorBlockKind::SoapMean
            | DescriptorBlockKind::SoapVariance
            | DescriptorBlockKind::AceNu3Mean
            | DescriptorBlockKind::SoapLeftover
            | DescriptorBlockKind::ProviderFeature => {
                return Err(DescriptorError::UniversalGeometryRequired);
            }
        };
        if let Some(index) = aggregated.iter().position(|value| !value.is_finite()) {
            return Err(DescriptorError::NonFiniteDescriptor {
                block: block_index,
                index,
            });
        }
        let raw_norm = norm(&aggregated);
        let denominator = contractive_denominator(raw_norm);
        for value in &mut aggregated {
            *value /= denominator;
        }
        let offset = values.len();
        let len = aggregated.len();
        values.extend(aggregated);
        metadata.push(DescriptorBlockMetadata {
            kind: block.kind(),
            n_max: block.n_max(),
            l_max: block.l_max(),
            cutoff: block.cutoff(),
            offset,
            len,
            raw_norm,
            normalization: CONTRACTIVE_L2_NORMALIZATION_SCHEMA.into(),
        });
    }
    Ok(DescriptorVector {
        schema_name: schema.name.clone(),
        schema_version: schema.version,
        values,
        blocks: metadata,
        provider_identity: None,
    })
}

pub(super) fn describe_local(
    schema: &DescriptorSchema,
    geometry: DescriptorGeometry,
    coordinates: ArrayView1<f64>,
    species: Option<&[u32]>,
) -> Result<Array2<f64>, DescriptorError> {
    let maximum_cutoff = schema
        .blocks
        .iter()
        .map(|block| block.cutoff())
        .fold(0.0, f64::max);
    let environment = Environment::new(geometry, coordinates, species, maximum_cutoff)?;
    let atoms = environment.atoms;
    let mut blocks = Vec::with_capacity(schema.blocks.len() + 1);
    let mut centres = Array2::zeros((atoms, SPECIES_CHANNELS));
    for atom in 0..atoms {
        let sketch = species_sketch(environment.species[atom]);
        for (column, value) in sketch.into_iter().enumerate() {
            centres[[atom, column]] = value;
        }
    }
    normalize_local_block(&mut centres);
    blocks.push(centres);
    for (block_index, block) in schema.blocks.iter().copied().enumerate() {
        let mut local = match block.kind() {
            DescriptorBlockKind::PairRadial => local_pair_radial(&environment, block),
            DescriptorBlockKind::ThreeBodyAngular => local_three_body_angular(&environment, block),
            DescriptorBlockKind::GraphTopology => local_graph_topology(&environment, block),
            DescriptorBlockKind::InvariantSoapMean => {
                local_invariant_spectrum(&environment, block, false)
            }
            DescriptorBlockKind::InvariantAceNu3Mean => {
                local_invariant_spectrum(&environment, block, true)
            }
            DescriptorBlockKind::ChiralMoment => local_chiral_moments(&environment, block),
            DescriptorBlockKind::SoapMean
            | DescriptorBlockKind::SoapVariance
            | DescriptorBlockKind::AceNu3Mean
            | DescriptorBlockKind::SoapLeftover
            | DescriptorBlockKind::ProviderFeature => {
                return Err(DescriptorError::UniversalGeometryRequired);
            }
        };
        if let Some(index) = local.iter().position(|value| !value.is_finite()) {
            return Err(DescriptorError::NonFiniteDescriptor {
                block: block_index,
                index,
            });
        }
        normalize_local_block(&mut local);
        blocks.push(local);
    }

    let columns = blocks.iter().map(|block| block.ncols()).sum();
    let block_scale = 1.0 / (blocks.len() as f64).sqrt();
    let mut output = Array2::zeros((atoms, columns));
    let mut offset = 0;
    for block in blocks {
        for atom in 0..atoms {
            for column in 0..block.ncols() {
                output[[atom, offset + column]] = block_scale * block[[atom, column]];
            }
        }
        offset += block.ncols();
    }
    Ok(output)
}

fn normalize_local_block(block: &mut Array2<f64>) {
    for mut row in block.rows_mut() {
        let raw_norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        let denominator = contractive_denominator(raw_norm);
        for value in &mut row {
            *value /= denominator;
        }
    }
}

/// Unit-scale radial contraction for dimensionless invariant blocks.
///
/// The map `x -> x / sqrt(1 + ||x||^2)` has norm below one and Jacobian
/// spectral norm at most one. It is therefore nonexpansive, retains amplitude
/// near zero, and cannot promote cancellation-scale signals to unit vectors.
fn contractive_denominator(raw_norm: f64) -> f64 {
    raw_norm.hypot(1.0)
}

struct Environment {
    atoms: usize,
    species: Vec<u32>,
    neighbors: Vec<Vec<NeighborImage>>,
}

#[derive(Debug, Clone, Copy)]
struct NeighborImage {
    atom: usize,
    displacement: [f64; 3],
    distance: f64,
}

impl Environment {
    fn new(
        geometry: DescriptorGeometry,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
        maximum_cutoff: f64,
    ) -> Result<Self, DescriptorError> {
        let atoms = coordinates.len() / 3;
        let positions = (0..atoms)
            .map(|atom| {
                [
                    coordinates[3 * atom],
                    coordinates[3 * atom + 1],
                    coordinates[3 * atom + 2],
                ]
            })
            .collect::<Vec<_>>();
        let periodic_cell = if geometry.periodic.iter().any(|&axis| axis) {
            let cell = geometry.cell.expect("fully periodic geometry has a cell");
            Some(
                Cell::from_vectors(
                    [cell[0], cell[1], cell[2]],
                    [cell[3], cell[4], cell[5]],
                    [cell[6], cell[7], cell[8]],
                    [0.0; 3],
                )
                .map_err(|_| DescriptorError::NeighborSearch)?,
            )
        } else {
            None
        };
        let image_bounds = periodic_cell
            .as_ref()
            .map(|cell| periodic_image_bounds(cell, geometry, maximum_cutoff))
            .transpose()?
            .unwrap_or([0; 3]);
        let mut neighbors = vec![Vec::new(); atoms];
        for centre in 0..atoms {
            for target in 0..atoms {
                let delta = [
                    positions[target][0] - positions[centre][0],
                    positions[target][1] - positions[centre][1],
                    positions[target][2] - positions[centre][2],
                ];
                let base = geometry.displacement(delta);
                for na in -image_bounds[0]..=image_bounds[0] {
                    for nb in -image_bounds[1]..=image_bounds[1] {
                        for nc in -image_bounds[2]..=image_bounds[2] {
                            if centre == target && na == 0 && nb == 0 && nc == 0 {
                                continue;
                            }
                            let shift = periodic_cell
                                .as_ref()
                                .map(|cell| cell.lattice_shift(na, nb, nc))
                                .unwrap_or([0.0; 3]);
                            let displacement = [
                                base[0] + shift[0] / geometry.length_scale,
                                base[1] + shift[1] / geometry.length_scale,
                                base[2] + shift[2] / geometry.length_scale,
                            ];
                            let distance = dot(displacement, displacement).sqrt();
                            if distance <= 1e-12 || distance >= maximum_cutoff {
                                continue;
                            }
                            neighbors[centre].push(NeighborImage {
                                atom: target,
                                displacement,
                                distance,
                            });
                        }
                    }
                }
            }
            neighbors[centre].sort_by(|left, right| {
                left.distance
                    .total_cmp(&right.distance)
                    .then_with(|| left.atom.cmp(&right.atom))
                    .then_with(|| left.displacement[0].total_cmp(&right.displacement[0]))
                    .then_with(|| left.displacement[1].total_cmp(&right.displacement[1]))
                    .then_with(|| left.displacement[2].total_cmp(&right.displacement[2]))
            });
        }
        Ok(Self {
            atoms,
            species: species
                .map(<[u32]>::to_vec)
                .unwrap_or_else(|| vec![0; atoms]),
            neighbors,
        })
    }
}

fn periodic_image_bounds(
    cell: &Cell,
    geometry: DescriptorGeometry,
    maximum_cutoff: f64,
) -> Result<[i32; 3], DescriptorError> {
    let physical_cutoff = maximum_cutoff * geometry.length_scale;
    let widths = cell.widths();
    let mut bounds = [0; 3];
    for axis in 0..3 {
        if !geometry.periodic[axis] {
            continue;
        }
        let bound = (physical_cutoff / widths[axis] + 0.5).ceil();
        if !bound.is_finite() || bound > i32::MAX as f64 {
            return Err(DescriptorError::NeighborSearch);
        }
        bounds[axis] = bound as i32;
    }
    Ok(bounds)
}

fn pair_radial(environment: &Environment, block: DescriptorBlockSpec) -> Vec<f64> {
    let radial_bins = block.n_max();
    let cutoff = block.cutoff();
    let width = cutoff / radial_bins as f64;
    let mut spectrum = vec![0.0; radial_bins * PAIR_CHANNELS];
    let centre_weight = 1.0 / environment.atoms.max(1) as f64;
    for centre in 0..environment.atoms {
        for neighbor in &environment.neighbors[centre] {
            let distance = neighbor.distance;
            if distance <= 1e-12 || distance >= cutoff {
                continue;
            }
            let cutoff_weight = cosine_cutoff(distance, cutoff);
            let species = pair_species(
                environment.species[centre],
                environment.species[neighbor.atom],
            );
            for radial in 0..radial_bins {
                let centre = (radial as f64 + 0.5) * width;
                let scaled = (distance - centre) / (0.65 * width);
                let basis = (-0.5 * scaled * scaled).exp() * cutoff_weight * centre_weight;
                for channel in 0..PAIR_CHANNELS {
                    spectrum[channel * radial_bins + radial] += basis * species[channel];
                }
            }
        }
    }
    spectrum
}

fn local_pair_radial(environment: &Environment, block: DescriptorBlockSpec) -> Array2<f64> {
    let radial_bins = block.n_max();
    let cutoff = block.cutoff();
    let width = cutoff / radial_bins as f64;
    let mut spectrum = Array2::zeros((environment.atoms, radial_bins * PAIR_CHANNELS));
    for centre in 0..environment.atoms {
        for neighbor in &environment.neighbors[centre] {
            let distance = neighbor.distance;
            if distance <= 1e-12 || distance >= cutoff {
                continue;
            }
            let cutoff_weight = cosine_cutoff(distance, cutoff);
            let species = pair_species(
                environment.species[centre],
                environment.species[neighbor.atom],
            );
            for radial in 0..radial_bins {
                let shell = (radial as f64 + 0.5) * width;
                let scaled = (distance - shell) / (0.65 * width);
                let basis = (-0.5 * scaled * scaled).exp() * cutoff_weight;
                for channel in 0..PAIR_CHANNELS {
                    spectrum[[centre, channel * radial_bins + radial]] += basis * species[channel];
                }
            }
        }
    }
    spectrum
}

fn three_body_angular(environment: &Environment, block: DescriptorBlockSpec) -> Vec<f64> {
    let radial_bins = block.n_max();
    let angular_orders = block.l_max() + 1;
    let cutoff = block.cutoff();
    let width = cutoff / radial_bins as f64;
    let mut spectrum = vec![0.0; radial_bins * angular_orders * TRIPLE_CHANNELS];
    for centre in 0..environment.atoms {
        let neighbors = environment.neighbors[centre]
            .iter()
            .filter(|neighbor| neighbor.distance < cutoff)
            .collect::<Vec<_>>();
        for left_index in 0..neighbors.len() {
            let left = neighbors[left_index];
            let left_distance = left.distance;
            if left_distance <= 1e-12 || left_distance >= cutoff {
                continue;
            }
            for &right in neighbors.iter().skip(left_index + 1) {
                let right_distance = right.distance;
                if right_distance <= 1e-12 || right_distance >= cutoff {
                    continue;
                }
                let left_vector = left.displacement;
                let right_vector = right.displacement;
                let cosine = (dot(left_vector, right_vector) / (left_distance * right_distance))
                    .clamp(-1.0, 1.0);
                let angular = legendre_values(cosine, block.l_max());
                let mean_distance = 0.5 * (left_distance + right_distance);
                let cutoff_weight =
                    cosine_cutoff(left_distance, cutoff) * cosine_cutoff(right_distance, cutoff);
                let species = triple_species(
                    environment.species[centre],
                    environment.species[left.atom],
                    environment.species[right.atom],
                );
                for radial in 0..radial_bins {
                    let shell = (radial as f64 + 0.5) * width;
                    let scaled = (mean_distance - shell) / (0.7 * width);
                    let radial_weight = (-0.5 * scaled * scaled).exp() * cutoff_weight;
                    for (order, &angular_value) in angular.iter().enumerate() {
                        for channel in 0..TRIPLE_CHANNELS {
                            let index = (channel * radial_bins + radial) * angular_orders + order;
                            spectrum[index] += radial_weight * angular_value * species[channel]
                                / environment.atoms.max(1) as f64;
                        }
                    }
                }
            }
        }
    }
    spectrum
}

fn local_three_body_angular(environment: &Environment, block: DescriptorBlockSpec) -> Array2<f64> {
    let radial_bins = block.n_max();
    let angular_orders = block.l_max() + 1;
    let cutoff = block.cutoff();
    let width = cutoff / radial_bins as f64;
    let mut spectrum = Array2::zeros((
        environment.atoms,
        radial_bins * angular_orders * TRIPLE_CHANNELS,
    ));
    for centre in 0..environment.atoms {
        let neighbors = environment.neighbors[centre]
            .iter()
            .filter(|neighbor| neighbor.distance < cutoff)
            .collect::<Vec<_>>();
        for left_index in 0..neighbors.len() {
            let left = neighbors[left_index];
            if left.distance <= 1e-12 || left.distance >= cutoff {
                continue;
            }
            for &right in neighbors.iter().skip(left_index + 1) {
                if right.distance <= 1e-12 || right.distance >= cutoff {
                    continue;
                }
                let cosine = (dot(left.displacement, right.displacement)
                    / (left.distance * right.distance))
                    .clamp(-1.0, 1.0);
                let angular = legendre_values(cosine, block.l_max());
                let mean_distance = 0.5 * (left.distance + right.distance);
                let cutoff_weight =
                    cosine_cutoff(left.distance, cutoff) * cosine_cutoff(right.distance, cutoff);
                let species = triple_species(
                    environment.species[centre],
                    environment.species[left.atom],
                    environment.species[right.atom],
                );
                for radial in 0..radial_bins {
                    let shell = (radial as f64 + 0.5) * width;
                    let scaled = (mean_distance - shell) / (0.7 * width);
                    let radial_weight = (-0.5 * scaled * scaled).exp() * cutoff_weight;
                    for (order, &angular_value) in angular.iter().enumerate() {
                        for channel in 0..TRIPLE_CHANNELS {
                            let index = (channel * radial_bins + radial) * angular_orders + order;
                            spectrum[[centre, index]] +=
                                radial_weight * angular_value * species[channel];
                        }
                    }
                }
            }
        }
    }
    spectrum
}

fn graph_topology(environment: &Environment, block: DescriptorBlockSpec) -> Vec<f64> {
    let scales = block.n_max();
    let mut spectrum = Vec::with_capacity(scales * GRAPH_FEATURES);
    for scale in 1..=scales {
        let threshold = block.cutoff() * scale as f64 / scales as f64;
        let mut averaged = [0.0; GRAPH_FEATURES];
        for centre in 0..environment.atoms {
            let local = graph_features(environment, centre, threshold);
            for (value, contribution) in averaged.iter_mut().zip(local) {
                *value += contribution / environment.atoms.max(1) as f64;
            }
        }
        spectrum.extend(averaged);
    }
    spectrum
}

fn local_graph_topology(environment: &Environment, block: DescriptorBlockSpec) -> Array2<f64> {
    let scales = block.n_max();
    let mut spectrum = Array2::zeros((environment.atoms, scales * GRAPH_FEATURES));
    for centre in 0..environment.atoms {
        for scale in 1..=scales {
            let threshold = block.cutoff() * scale as f64 / scales as f64;
            let local = graph_features(environment, centre, threshold);
            for (feature, value) in local.into_iter().enumerate() {
                spectrum[[centre, (scale - 1) * GRAPH_FEATURES + feature]] = value;
            }
        }
    }
    spectrum
}

fn graph_features(
    environment: &Environment,
    centre: usize,
    threshold: f64,
) -> [f64; GRAPH_FEATURES] {
    let neighbors = environment.neighbors[centre]
        .iter()
        .filter(|neighbor| neighbor.distance < threshold)
        .collect::<Vec<_>>();
    let count = neighbors.len();
    if count == 0 {
        return [0.0; GRAPH_FEATURES];
    }
    let centre_weights = neighbors
        .iter()
        .map(|neighbor| cosine_cutoff(neighbor.distance, threshold))
        .collect::<Vec<_>>();
    let mut adjacency = vec![0.0; count * count];
    for left in 0..count {
        for right in left + 1..count {
            let displacement = [
                neighbors[right].displacement[0] - neighbors[left].displacement[0],
                neighbors[right].displacement[1] - neighbors[left].displacement[1],
                neighbors[right].displacement[2] - neighbors[left].displacement[2],
            ];
            let distance = dot(displacement, displacement).sqrt();
            if distance >= threshold {
                continue;
            }
            let weight = cosine_cutoff(distance, threshold);
            adjacency[left * count + right] = weight;
            adjacency[right * count + left] = weight;
        }
    }
    let degree = centre_weights.iter().sum::<f64>();
    let second_moment = centre_weights
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>();
    let mut row_two = vec![0.0; count];
    let mut neighbor_degree_moment = 0.0;
    for neighbor in 0..count {
        let graph_degree = centre_weights[neighbor]
            + (0..count)
                .map(|other| adjacency[neighbor * count + other])
                .sum::<f64>();
        neighbor_degree_moment += centre_weights[neighbor] * graph_degree;
        row_two[neighbor] = (0..count)
            .map(|other| centre_weights[other] * adjacency[other * count + neighbor])
            .sum::<f64>();
    }
    let third_moment = centre_weights
        .iter()
        .zip(&row_two)
        .map(|(weight, return_walk)| weight * return_walk)
        .sum::<f64>();
    let fourth_moment =
        second_moment * second_moment + row_two.iter().map(|value| value * value).sum::<f64>();
    let radial_moment = neighbors
        .iter()
        .zip(&centre_weights)
        .map(|(neighbor, weight)| weight * neighbor.distance / threshold)
        .sum::<f64>();
    let rank_affinity = centre_weights
        .iter()
        .take(12)
        .enumerate()
        .map(|(rank, weight)| weight / (rank + 1) as f64)
        .sum::<f64>();
    let centre_species = species_scalar(environment.species[centre]);
    let species_degree = neighbors
        .iter()
        .zip(&centre_weights)
        .map(|(neighbor, weight)| weight * species_scalar(environment.species[neighbor.atom]))
        .sum::<f64>();
    let species_contrast = neighbors
        .iter()
        .zip(&centre_weights)
        .map(|(neighbor, weight)| {
            weight * (centre_species - species_scalar(environment.species[neighbor.atom])).abs()
        })
        .sum::<f64>();
    [
        degree,
        second_moment,
        third_moment,
        fourth_moment,
        neighbor_degree_moment,
        (degree * degree - second_moment).max(0.0),
        radial_moment,
        rank_affinity,
        species_degree,
        species_contrast,
    ]
}

fn invariant_spectrum(
    environment: &Environment,
    block: DescriptorBlockSpec,
    ace: bool,
) -> Vec<f64> {
    let per_channel = if ace {
        crate::ace::dim(block.n_max(), block.l_max())
    } else {
        block.n_max() * (block.n_max() + 1) / 2 * (block.l_max() + 1)
    };
    let mut spectrum = vec![0.0; per_channel * SPECIES_CHANNELS];
    for centre in 0..environment.atoms {
        for channel in 0..SPECIES_CHANNELS {
            let neighbors = environment.neighbors[centre]
                .iter()
                .filter(|neighbor| neighbor.distance < block.cutoff())
                .map(|neighbor| {
                    (
                        neighbor.displacement,
                        species_sketch(environment.species[neighbor.atom])[channel],
                    )
                })
                .collect::<Vec<_>>();
            let (soap, coefficients) = central_spectrum_from_displacements(&neighbors, block.soap);
            let local = if ace {
                crate::ace::from_c(&coefficients, block.n_max(), block.l_max())
            } else {
                soap
            };
            for (index, value) in local.into_iter().enumerate() {
                spectrum[channel * per_channel + index] += value / environment.atoms.max(1) as f64;
            }
        }
    }
    spectrum
}

fn local_invariant_spectrum(
    environment: &Environment,
    block: DescriptorBlockSpec,
    ace: bool,
) -> Array2<f64> {
    let per_channel = if ace {
        crate::ace::dim(block.n_max(), block.l_max())
    } else {
        block.n_max() * (block.n_max() + 1) / 2 * (block.l_max() + 1)
    };
    let mut spectrum = Array2::zeros((environment.atoms, per_channel * SPECIES_CHANNELS));
    for centre in 0..environment.atoms {
        for channel in 0..SPECIES_CHANNELS {
            let neighbors = environment.neighbors[centre]
                .iter()
                .filter(|neighbor| neighbor.distance < block.cutoff())
                .map(|neighbor| {
                    (
                        neighbor.displacement,
                        species_sketch(environment.species[neighbor.atom])[channel],
                    )
                })
                .collect::<Vec<_>>();
            let (soap, coefficients) = central_spectrum_from_displacements(&neighbors, block.soap);
            let local = if ace {
                crate::ace::from_c(&coefficients, block.n_max(), block.l_max())
            } else {
                soap
            };
            for (index, value) in local.into_iter().enumerate() {
                spectrum[[centre, channel * per_channel + index]] += value;
            }
        }
    }
    spectrum
}

fn chiral_moments(environment: &Environment, block: DescriptorBlockSpec) -> Vec<f64> {
    let radial_channels = block.n_max();
    let moment_channels = radial_channels * SPECIES_CHANNELS;
    let triple_count = moment_channels
        .saturating_mul(moment_channels.saturating_sub(1))
        .saturating_mul(moment_channels.saturating_sub(2))
        / 6;
    let mut spectrum = vec![0.0; triple_count * SPECIES_CHANNELS];
    let radial_width = 1.0 / radial_channels as f64;
    let centre_scale = 1.0 / environment.atoms.max(1) as f64;
    for centre in 0..environment.atoms {
        let mut moments = vec![[0.0; 3]; moment_channels];
        for neighbor in environment.neighbors[centre]
            .iter()
            .filter(|neighbor| neighbor.distance < block.cutoff())
        {
            let direction = [
                neighbor.displacement[0] / neighbor.distance,
                neighbor.displacement[1] / neighbor.distance,
                neighbor.displacement[2] / neighbor.distance,
            ];
            let reduced_distance = neighbor.distance / block.cutoff();
            let envelope = cosine_cutoff(neighbor.distance, block.cutoff());
            let species = species_sketch(environment.species[neighbor.atom]);
            for radial in 0..radial_channels {
                let radial_centre = (radial as f64 + 0.5) * radial_width;
                let scaled = (reduced_distance - radial_centre) / (0.65 * radial_width);
                let radial_weight = (-0.5 * scaled * scaled).exp() * envelope;
                for channel in 0..SPECIES_CHANNELS {
                    let weight = radial_weight * species[channel];
                    let moment = &mut moments[radial * SPECIES_CHANNELS + channel];
                    for axis in 0..3 {
                        moment[axis] += weight * direction[axis];
                    }
                }
            }
        }
        let centre_species = species_sketch(environment.species[centre]);
        let mut triple = 0;
        for left in 0..moment_channels {
            for middle in left + 1..moment_channels {
                for right in middle + 1..moment_channels {
                    let pseudoscalar = dot(moments[left], cross(moments[middle], moments[right]));
                    for channel in 0..SPECIES_CHANNELS {
                        spectrum[channel * triple_count + triple] +=
                            centre_scale * centre_species[channel] * pseudoscalar;
                    }
                    triple += 1;
                }
            }
        }
    }
    spectrum
}

fn local_chiral_moments(environment: &Environment, block: DescriptorBlockSpec) -> Array2<f64> {
    let radial_channels = block.n_max();
    let moment_channels = radial_channels * SPECIES_CHANNELS;
    let triple_count = moment_channels
        .saturating_mul(moment_channels.saturating_sub(1))
        .saturating_mul(moment_channels.saturating_sub(2))
        / 6;
    let mut spectrum = Array2::zeros((environment.atoms, triple_count * SPECIES_CHANNELS));
    let radial_width = 1.0 / radial_channels as f64;
    for centre in 0..environment.atoms {
        let mut moments = vec![[0.0; 3]; moment_channels];
        for neighbor in environment.neighbors[centre]
            .iter()
            .filter(|neighbor| neighbor.distance < block.cutoff())
        {
            let direction = [
                neighbor.displacement[0] / neighbor.distance,
                neighbor.displacement[1] / neighbor.distance,
                neighbor.displacement[2] / neighbor.distance,
            ];
            let reduced_distance = neighbor.distance / block.cutoff();
            let envelope = cosine_cutoff(neighbor.distance, block.cutoff());
            let species = species_sketch(environment.species[neighbor.atom]);
            for radial in 0..radial_channels {
                let radial_centre = (radial as f64 + 0.5) * radial_width;
                let scaled = (reduced_distance - radial_centre) / (0.65 * radial_width);
                let radial_weight = (-0.5 * scaled * scaled).exp() * envelope;
                for channel in 0..SPECIES_CHANNELS {
                    let weight = radial_weight * species[channel];
                    let moment = &mut moments[radial * SPECIES_CHANNELS + channel];
                    for axis in 0..3 {
                        moment[axis] += weight * direction[axis];
                    }
                }
            }
        }
        let centre_species = species_sketch(environment.species[centre]);
        let mut triple = 0;
        for left in 0..moment_channels {
            for middle in left + 1..moment_channels {
                for right in middle + 1..moment_channels {
                    let pseudoscalar = dot(moments[left], cross(moments[middle], moments[right]));
                    for channel in 0..SPECIES_CHANNELS {
                        spectrum[[centre, channel * triple_count + triple]] =
                            centre_species[channel] * pseudoscalar;
                    }
                    triple += 1;
                }
            }
        }
    }
    spectrum
}

fn pair_species(left: u32, right: u32) -> [f64; PAIR_CHANNELS] {
    let left = species_scalar(left);
    let right = species_scalar(right);
    [
        1.0,
        0.5 * (left + right),
        (left - right).abs(),
        left * right,
    ]
}

fn triple_species(centre: u32, left: u32, right: u32) -> [f64; TRIPLE_CHANNELS] {
    let centre = species_scalar(centre);
    let left = species_scalar(left);
    let right = species_scalar(right);
    [
        1.0,
        (centre + left + right) / 3.0,
        centre * 0.5 * (left + right),
    ]
}

fn species_sketch(species: u32) -> [f64; SPECIES_CHANNELS] {
    let atomic = species_scalar(species);
    [
        1.0,
        atomic,
        (species as f64 * 0.618_033_988_749_894_8 * TAU).cos(),
    ]
}

fn species_scalar(species: u32) -> f64 {
    let species = species as f64;
    species / (species + 8.0)
}

fn legendre_values(cosine: f64, maximum: usize) -> Vec<f64> {
    let mut values = vec![0.0; maximum + 1];
    values[0] = 1.0;
    if maximum == 0 {
        return values;
    }
    values[1] = cosine;
    for order in 2..=maximum {
        values[order] = ((2 * order - 1) as f64 * cosine * values[order - 1]
            - (order - 1) as f64 * values[order - 2])
            / order as f64;
    }
    values
}

fn cosine_cutoff(distance: f64, cutoff: f64) -> f64 {
    let phase = 0.5 * PI * (1.0 - distance / cutoff).clamp(0.0, 1.0);
    phase.sin().powi(2)
}

fn determinant(cell: [f64; 9]) -> f64 {
    dot(
        [cell[0], cell[1], cell[2]],
        cross([cell[3], cell[4], cell[5]], [cell[6], cell[7], cell[8]]),
    )
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
