//! Fixed-dimensional invariant geometry for cross-system basin catalogues.

use super::{
    DescriptorBlockKind, DescriptorBlockMetadata, DescriptorBlockSpec, DescriptorError,
    DescriptorSchema, DescriptorSpace, DescriptorVector,
};
use crate::soap::central_spectrum_from_displacements;
use linkcell::{Cell, knearest};
use ndarray::ArrayView1;
use std::f64::consts::{PI, TAU};

/// Stable schema name for the cross-system descriptor.
pub const UNIVERSAL_DESCRIPTOR_SCHEMA: &str = "anneal-universal-pes";
/// Stable schema version for the cross-system descriptor.
pub const UNIVERSAL_DESCRIPTOR_VERSION: u32 = 1;

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
        if let Some(cell) = cell {
            if cell.iter().any(|value| !value.is_finite()) || determinant(cell).abs() <= 1e-12 {
                return Err(DescriptorError::InvalidCell);
            }
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

    fn displacement(self, delta: [f64; 3]) -> [f64; 3] {
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
    let environment = Environment::new(geometry, coordinates, species)?;
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
            DescriptorBlockKind::SoapMean
            | DescriptorBlockKind::SoapVariance
            | DescriptorBlockKind::AceNu3Mean
            | DescriptorBlockKind::SoapLeftover => {
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
        if raw_norm > 0.0 {
            for value in &mut aggregated {
                *value /= raw_norm;
            }
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
        });
    }
    Ok(DescriptorVector {
        schema_name: schema.name.clone(),
        schema_version: schema.version,
        values,
        blocks: metadata,
    })
}

struct Environment {
    atoms: usize,
    displacements: Vec<[f64; 3]>,
    distances: Vec<f64>,
    species: Vec<u32>,
    nearest: Vec<Vec<usize>>,
}

impl Environment {
    fn new(
        geometry: DescriptorGeometry,
        coordinates: ArrayView1<f64>,
        species: Option<&[u32]>,
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
        let periodic_cell = if geometry.periodic == [true; 3] {
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
        let mut displacements = vec![[0.0; 3]; atoms * atoms];
        let mut distances = vec![0.0; atoms * atoms];
        for i in 0..atoms {
            for j in i + 1..atoms {
                let displacement = match periodic_cell.as_ref() {
                    Some(cell) => linkcell_displacement(cell, positions[i], positions[j], geometry),
                    None => geometry.displacement([
                        positions[j][0] - positions[i][0],
                        positions[j][1] - positions[i][1],
                        positions[j][2] - positions[i][2],
                    ]),
                };
                let reverse = [-displacement[0], -displacement[1], -displacement[2]];
                let distance = dot(displacement, displacement).sqrt();
                displacements[i * atoms + j] = displacement;
                displacements[j * atoms + i] = reverse;
                distances[i * atoms + j] = distance;
                distances[j * atoms + i] = distance;
            }
        }
        let nearest_count = atoms.saturating_sub(1).min(12);
        let nearest = match periodic_cell.as_ref() {
            Some(cell) if nearest_count > 0 => knearest(
                &positions,
                cell,
                nearest_count,
                None,
                Some(geometry.length_scale),
            )
            .map_err(|_| DescriptorError::NeighborSearch)?
            .into_iter()
            .map(|neighbors| neighbors.indices)
            .collect(),
            _ => nearest_from_distances(&distances, atoms, nearest_count),
        };
        Ok(Self {
            atoms,
            displacements,
            distances,
            species: species
                .map(<[u32]>::to_vec)
                .unwrap_or_else(|| vec![0; atoms]),
            nearest,
        })
    }

    fn displacement(&self, i: usize, j: usize) -> [f64; 3] {
        self.displacements[i * self.atoms + j]
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        self.distances[i * self.atoms + j]
    }
}

fn pair_radial(environment: &Environment, block: DescriptorBlockSpec) -> Vec<f64> {
    let radial_bins = block.n_max();
    let cutoff = block.cutoff();
    let width = cutoff / radial_bins as f64;
    let mut spectrum = vec![0.0; radial_bins * PAIR_CHANNELS];
    for i in 0..environment.atoms {
        for j in i + 1..environment.atoms {
            let distance = environment.distance(i, j);
            if distance <= 1e-12 || distance >= cutoff {
                continue;
            }
            let cutoff_weight = cosine_cutoff(distance, cutoff);
            let species = pair_species(environment.species[i], environment.species[j]);
            for radial in 0..radial_bins {
                let centre = (radial as f64 + 0.5) * width;
                let scaled = (distance - centre) / (0.65 * width);
                let basis = (-0.5 * scaled * scaled).exp() * cutoff_weight;
                for channel in 0..PAIR_CHANNELS {
                    spectrum[channel * radial_bins + radial] += basis * species[channel];
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
        for left in 0..environment.atoms {
            if left == centre {
                continue;
            }
            let left_distance = environment.distance(centre, left);
            if left_distance <= 1e-12 || left_distance >= cutoff {
                continue;
            }
            for right in left + 1..environment.atoms {
                if right == centre {
                    continue;
                }
                let right_distance = environment.distance(centre, right);
                if right_distance <= 1e-12 || right_distance >= cutoff {
                    continue;
                }
                let left_vector = environment.displacement(centre, left);
                let right_vector = environment.displacement(centre, right);
                let cosine = (dot(left_vector, right_vector) / (left_distance * right_distance))
                    .clamp(-1.0, 1.0);
                let angular = legendre_values(cosine, block.l_max());
                let mean_distance = 0.5 * (left_distance + right_distance);
                let cutoff_weight =
                    cosine_cutoff(left_distance, cutoff) * cosine_cutoff(right_distance, cutoff);
                let species = triple_species(
                    environment.species[centre],
                    environment.species[left],
                    environment.species[right],
                );
                for radial in 0..radial_bins {
                    let shell = (radial as f64 + 0.5) * width;
                    let scaled = (mean_distance - shell) / (0.7 * width);
                    let radial_weight = (-0.5 * scaled * scaled).exp() * cutoff_weight;
                    for (order, &angular_value) in angular.iter().enumerate() {
                        for channel in 0..TRIPLE_CHANNELS {
                            let index = (channel * radial_bins + radial) * angular_orders + order;
                            spectrum[index] += radial_weight * angular_value * species[channel];
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
        spectrum.extend(graph_features(environment, threshold, 2 * scale));
    }
    spectrum
}

fn graph_features(
    environment: &Environment,
    threshold: f64,
    neighbor_rank: usize,
) -> [f64; GRAPH_FEATURES] {
    let atoms = environment.atoms;
    let mut adjacency = vec![false; atoms * atoms];
    let mut degree = vec![0usize; atoms];
    let mut edges = 0usize;
    let mut species_edge = 0.0;
    for i in 0..atoms {
        for j in i + 1..atoms {
            if environment.distance(i, j) < threshold {
                adjacency[i * atoms + j] = true;
                adjacency[j * atoms + i] = true;
                degree[i] += 1;
                degree[j] += 1;
                edges += 1;
                species_edge +=
                    species_scalar(environment.species[i]) * species_scalar(environment.species[j]);
            }
        }
    }
    let possible_edges = atoms.saturating_mul(atoms.saturating_sub(1)) / 2;
    let degree_scale = atoms.saturating_sub(1).max(1) as f64;
    let mean_degree = degree.iter().sum::<usize>() as f64 / atoms.max(1) as f64;
    let degree_variance = degree
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean_degree;
            delta * delta
        })
        .sum::<f64>()
        / atoms.max(1) as f64;
    let mut mutual_nearest_edges = 0usize;
    for i in 0..atoms {
        for &j in environment.nearest[i]
            .iter()
            .take(neighbor_rank.min(environment.nearest[i].len()))
        {
            if i < j
                && environment.nearest[j]
                    .iter()
                    .take(neighbor_rank.min(environment.nearest[j].len()))
                    .any(|&neighbor| neighbor == i)
            {
                mutual_nearest_edges += 1;
            }
        }
    }
    let isolated = degree.iter().filter(|&&value| value == 0).count();
    let mut triangles = 0usize;
    for i in 0..atoms {
        for j in i + 1..atoms {
            if !adjacency[i * atoms + j] {
                continue;
            }
            for k in j + 1..atoms {
                if adjacency[i * atoms + k] && adjacency[j * atoms + k] {
                    triangles += 1;
                }
            }
        }
    }
    let triples = atoms
        .saturating_mul(atoms.saturating_sub(1))
        .saturating_mul(atoms.saturating_sub(2))
        / 6;
    let wedges = degree
        .iter()
        .map(|&value| value.saturating_mul(value.saturating_sub(1)) / 2)
        .sum::<usize>();
    let mut fourth_moment = 0.0;
    for i in 0..atoms {
        for j in 0..atoms {
            let common = (0..atoms)
                .filter(|&k| adjacency[i * atoms + k] && adjacency[k * atoms + j])
                .count() as f64;
            fourth_moment += common * common;
        }
    }
    let (components, largest_component) = component_summary(&adjacency, atoms);
    [
        ratio(edges, possible_edges),
        mean_degree / degree_scale,
        degree_variance.sqrt() / degree_scale,
        ratio(mutual_nearest_edges, possible_edges),
        ratio(isolated, atoms),
        ratio(triangles, triples),
        if wedges == 0 {
            0.0
        } else {
            3.0 * triangles as f64 / wedges as f64
        },
        fourth_moment / (atoms.max(1) as f64 * degree_scale.powi(4)),
        ratio(components, atoms),
        if edges == 0 {
            0.0
        } else {
            0.5 * (ratio(largest_component, atoms) + species_edge / edges as f64)
        },
    ]
}

fn linkcell_displacement(
    cell: &Cell,
    left: [f64; 3],
    right: [f64; 3],
    geometry: DescriptorGeometry,
) -> [f64; 3] {
    let left_fractional = cell.fractional(left);
    let right_fractional = cell.fractional(right);
    let mut displacement = [0.0; 3];
    for axis in 0..3 {
        displacement[axis] = right_fractional[axis] - left_fractional[axis];
        displacement[axis] -= displacement[axis].round();
    }
    let cartesian = cell.cartesian(displacement);
    [
        cartesian[0] / geometry.length_scale,
        cartesian[1] / geometry.length_scale,
        cartesian[2] / geometry.length_scale,
    ]
}

fn nearest_from_distances(distances: &[f64], atoms: usize, count: usize) -> Vec<Vec<usize>> {
    (0..atoms)
        .map(|source| {
            let mut neighbors = (0..atoms)
                .filter(|&candidate| candidate != source)
                .map(|candidate| (distances[source * atoms + candidate], candidate))
                .collect::<Vec<_>>();
            neighbors.sort_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            neighbors
                .into_iter()
                .take(count)
                .map(|(_, index)| index)
                .collect()
        })
        .collect()
}

fn component_summary(adjacency: &[bool], atoms: usize) -> (usize, usize) {
    let mut seen = vec![false; atoms];
    let mut components = 0usize;
    let mut largest = 0usize;
    for root in 0..atoms {
        if seen[root] {
            continue;
        }
        components += 1;
        seen[root] = true;
        let mut stack = vec![root];
        let mut size = 0usize;
        while let Some(vertex) = stack.pop() {
            size += 1;
            for neighbor in 0..atoms {
                if adjacency[vertex * atoms + neighbor] && !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        largest = largest.max(size);
    }
    (components, largest)
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
            let mut neighbors = Vec::with_capacity(environment.atoms.saturating_sub(1));
            for neighbor in 0..environment.atoms {
                if neighbor == centre {
                    continue;
                }
                neighbors.push((
                    environment.displacement(centre, neighbor),
                    species_sketch(environment.species[neighbor])[channel],
                ));
            }
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
    0.5 * (1.0 + (PI * distance / cutoff).cos())
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

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
