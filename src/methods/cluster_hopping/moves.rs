//! Cluster proposal libraries and scale-aware move kernels.

use super::preset::LennardJonesPreset;
use super::*;

/// Exactly one proposal library selected by a cluster preset.
///
/// The adaptive allocator may still choose among the kernels inside the
/// selected library.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveLibrary {
    /// Wales-Doye atomic moves.
    Atomic,
    /// Productive atomic arms only.
    Lean,
    /// Productive atomic arms plus composed surface relocation.
    LeanBurst,
    /// Atomic moves plus the heavy-tailed visiting kernel.
    Visit,
    /// Atomic moves plus twinning.
    Twin,
    /// Regrow from the incumbent's observed local order.
    SelfReseed,
    /// Let a posterior choose the observed-order construction.
    LearnedReseed,
    /// Offer every named and observed construction source.
    Reseed,
    /// Couple posterior-selected growth to twinning.
    GrowthAndTwin,
    /// Rigid molecular moves, optionally combined with atomic reactive moves.
    Molecular {
        /// Atom indices belonging to each declared rigid group.
        groups: Vec<Vec<usize>>,
        /// Keep atomic moves reachable for bond breaking and formation.
        reactive: bool,
    },
}

impl MoveLibrary {
    /// Declared rigid groups for a molecular library.
    pub fn declared_groups(&self) -> Option<&[Vec<usize>]> {
        match self {
            Self::Molecular { groups, .. } => Some(groups),
            _ => None,
        }
    }

    /// Builds proposal kernels using the scales declared by `cfg`.
    pub fn kernels(&self, cfg: &Config) -> Vec<ClusterMove> {
        let atomic = || {
            ClusterMove::library_scaled(
                cfg.n_points,
                cfg.length_scale,
                cfg.neighbour_cutoff,
                cfg.symmetrise_cutoff,
            )
        };
        let lean = || {
            ClusterMove::library_lean_scaled(
                cfg.n_points,
                cfg.length_scale,
                cfg.neighbour_cutoff,
                cfg.symmetrise_cutoff,
            )
        };
        match self {
            Self::Atomic => atomic(),
            Self::Lean => lean(),
            Self::LeanBurst => {
                let mut kernels = lean();
                kernels.push(ClusterMove::Burst {
                    n_points: cfg.n_points,
                    neighbour_cutoff: cfg.neighbour_cutoff,
                });
                if cfg.soap_mode != SoapProposalMode::Off {
                    kernels.push(soap_arm(cfg, None, None));
                }
                kernels
            }
            Self::Visit => {
                let mut kernels = atomic();
                kernels.push(ClusterMove::Visit { q_v: 2.7 });
                kernels
            }
            Self::Twin => {
                let mut kernels = atomic();
                kernels.push(ClusterMove::Twin {
                    n_points: cfg.n_points,
                });
                kernels
            }
            Self::SelfReseed | Self::LearnedReseed => {
                let mut kernels = atomic();
                kernels.push(ClusterMove::Reseed {
                    n_points: cfg.n_points,
                    source: crate::lattice::Source::Observed,
                });
                kernels
            }
            Self::Reseed => {
                let mut kernels = atomic();
                for source in crate::lattice::Source::library() {
                    kernels.push(ClusterMove::Reseed {
                        n_points: cfg.n_points,
                        source,
                    });
                }
                kernels
            }
            Self::GrowthAndTwin => {
                let mut kernels = atomic();
                kernels.push(ClusterMove::Reseed {
                    n_points: cfg.n_points,
                    source: crate::lattice::Source::Observed,
                });
                kernels.push(ClusterMove::Twin {
                    n_points: cfg.n_points,
                });
                kernels
            }
            Self::Molecular { groups, reactive } => {
                let mut kernels = if *reactive {
                    ClusterMove::library_combined_scaled(
                        cfg.n_points,
                        groups.clone(),
                        cfg.group_cutoff,
                        cfg.length_scale,
                    )
                } else {
                    ClusterMove::library_molecular_scaled(
                        groups.clone(),
                        cfg.group_cutoff,
                        cfg.length_scale,
                    )
                };
                let mobile: Vec<usize> = groups.iter().flatten().copied().collect();
                let mobile = if mobile.len() == cfg.n_points {
                    None
                } else {
                    Some(mobile)
                };
                if cfg.soap_mode != SoapProposalMode::Off {
                    kernels.push(soap_arm(cfg, mobile, Some(groups.clone())));
                }
                kernels
            }
        }
    }

    /// Whether the library asks the construction posterior to select growth.
    pub fn learns_construction(&self) -> bool {
        matches!(self, Self::LearnedReseed | Self::GrowthAndTwin)
    }

    /// Whether the library contains rigid molecular proposal arms.
    pub fn is_molecular(&self) -> bool {
        matches!(self, Self::Molecular { .. })
    }

    /// Reactive setting carried by a molecular library.
    pub fn molecular_reactive(&self) -> Option<bool> {
        match self {
            Self::Molecular { reactive, .. } => Some(*reactive),
            _ => None,
        }
    }
}

/// The move library, dispatched by value.
///
/// [`MoveKernel::propose`] is generic over the generator, which makes the
/// trait not dyn compatible, so the library is an enum rather than a vector of
/// boxes. Keeping the generic parameter is worth more than boxing: it lets a
/// kernel be used with any generator without a virtual call per proposal.
pub enum ClusterMove {
    /// Displace every point uniformly: the standard basin-hopping move.
    AllPoints {
        /// Half-width of the per-coordinate displacement.
        step: f64,
    },
    /// Displace one point. Cheap and local, for polishing a packing.
    SinglePoint {
        /// Points in a state.
        n_points: usize,
        /// Half-width of the displacement.
        step: f64,
    },
    /// Displace every point by a draw from the Tsallis visiting distribution.
    ///
    /// The heavy-tailed proposal of generalised simulated annealing (Tsallis
    /// and Stariolo, doi:10.1016/S0378-4371(96)00271-3), which is where that
    /// method's power over classical annealing sits: the acceptance rule is a
    /// detail beside the visiting distribution. Most draws are small and a rare
    /// one is enormous, so the move produces its own large excursions without
    /// anyone choosing a step length, which is the property the rest of this
    /// library lacks. Every other kernel here is bounded by a scale set by
    /// hand.
    ///
    /// This matters because the crossing between funnels is a single
    /// perturbation followed by a single relaxation, not a walk: all 22
    /// crossings measured in 32 runs arrived in one improvement. The operator
    /// that crosses is this one, and it had no heavy tail available.
    ///
    /// Large draws are bounded by the container the driver already applies, so
    /// the tail reads as "scatter a point to the far side of the cluster"
    /// rather than as an unbounded coordinate.
    Visit {
        /// Tsallis visiting index; the literature default is 2.7.
        q_v: f64,
    },
    /// Several surface relocations composed into one proposal.
    ///
    /// Measured at 38 points, the crossing into the funnel that holds the
    /// answer completes from precursor structures the chain reaches through
    /// accepted moves *worse than its best*, and the draw that completes it is
    /// a single relocation. Run as separate hops, every intermediate state of
    /// that excursion must survive its own Metropolis test, so the excursion
    /// survives with the product of its acceptance probabilities. Composed
    /// into one proposal it pays one test on the final state only.
    ///
    /// `k` is geometric with mean about three, so most bursts are short and no
    /// scale is tuned. Each relocation acts on the structure the previous one
    /// produced, and nothing here reads any order parameter or morphology: the
    /// operator is the library's own relocation, repeated.
    Burst {
        /// Points in a state.
        n_points: usize,
        /// Separation below which two points count as neighbours.
        neighbour_cutoff: f64,
    },
    /// Rigidly relocate the least-bound group onto the cluster surface.
    ///
    /// The molecular analogue of the measured-productive operator. Atomic
    /// surface relocation tears a bonded molecule apart, so for molecular
    /// clusters the unit that moves is the group: the least-bound group,
    /// judged by its count of inter-group contacts, is translated to a
    /// random point on the cluster's surface shell and given a random rigid
    /// rotation about its own centroid. Intra-group geometry is preserved
    /// exactly, which is the constraint the quench would otherwise pay to
    /// restore, and nothing here reads species or morphology: the groups are
    /// the caller's declaration.
    GroupRelocate {
        /// Atom indices of each rigid group.
        groups: Vec<Vec<usize>>,
        /// Separation below which two atoms of different groups count as a
        /// contact.
        neighbour_cutoff: f64,
    },
    /// Several group relocations composed into one proposal: the burst
    /// analogue for molecular clusters, paying one acceptance test for a
    /// composed excursion.
    GroupBurst {
        /// Atom indices of each rigid group.
        groups: Vec<Vec<usize>>,
        /// Separation below which two atoms of different groups count as a
        /// contact.
        neighbour_cutoff: f64,
    },
    /// Small rigid displacement and rotation of one random group: the
    /// workhorse move of a molecular cluster, bond-preserving by
    /// construction where an atomic displacement pays the quench to restore
    /// every bond it stretched.
    GroupShake {
        /// Atom indices of each rigid group.
        groups: Vec<Vec<usize>>,
        /// Translation scale.
        amplitude: f64,
    },
    /// Relocate the least-coordinated point onto the surface.
    SurfaceRelocate(SurfaceRelocate),
    /// Rotate the outer shell against the core.
    ShellRotate(ShellRotate),
    /// Enforce an approximate rotational symmetry.
    Symmetrise(Symmetrise),
    /// Twin the structure across one of its dense planes.
    ///
    /// The move between a displacement, which never leaves the funnel, and a
    /// rebuild, which leaves it and lands far above the incumbent. Close
    /// packings differ by their stacking, and the operation relating them is a
    /// reflection in a dense plane: a decahedron is five tetrahedra sharing
    /// twin boundaries and an icosahedron is twenty. Reflecting one side
    /// leaves every neighbour relation on each side intact and changes only
    /// the contacts across the plane, so the proposal costs a boundary layer
    /// rather than a structure and can survive an acceptance test that a
    /// rebuilt candidate cannot.
    ///
    /// See [`crate::twin`].
    Twin {
        /// Points in a state.
        n_points: usize,
    },
    /// Rebuild the structure by growing a local order, and quench into it.
    ///
    /// The only move here that crosses a funnel boundary in one step. Every
    /// other proposal displaces points and relies on the quench to find a
    /// nearby minimum, so the basins a chain can reach from where it stands are
    /// the ones a displacement reaches, and no displacement of an icosahedral
    /// 98-point structure lands in the tetrahedral funnel with usable
    /// probability. A template is not reached, it is written down: the points
    /// are indistinguishable, so the family's sites *are* the proposal.
    ///
    /// Nothing about it is specific to one potential. The order to grow is
    /// read off the structure the chain stands on, by taking the neighbour
    /// offsets of its best-coordinated point, and the alternatives come from
    /// the classifier's template library rather than from a list of packings
    /// someone chose for this problem. Which is worth proposing is left to the
    /// allocator; nothing here knows that 38 points want a truncated
    /// octahedron.
    ///
    /// See [`crate::lattice`].
    Reseed {
        /// Points in a state.
        n_points: usize,
        /// Where the local order to grow comes from.
        source: crate::lattice::Source,
    },
    /// Wales and Doye's angular move on the worst-bound point.
    ///
    /// "Each angular displacement consisted of choosing random theta and phi
    /// spherical polar coordinates for the atom in question, taking the origin
    /// at the center of mass and replacing the radius with the maximum value in
    /// the cluster" (J. Phys. Chem. A 101, 5111).
    ///
    /// Not the same move as [`SurfaceRelocate`], which takes the
    /// least-coordinated point and places it near the surface. This takes the
    /// point with the highest pair energy and throws it to the far edge of the
    /// cluster at a random angle, which is a much larger step and is the move
    /// the 1997 paper used to reach the decahedral minima.
    Angular {
        /// Points in a state.
        n_points: usize,
        /// Lennard-Jones length scale used by the binding criterion.
        length_scale: f64,
        /// Lennard-Jones energy scale used by the binding criterion.
        energy_scale: f64,
    },
    /// Step in the SOAP power spectrum and pull back through \(J = \partial p/\partial R\).
    ///
    /// Concerted, not a one-atom hop. On a packing cluster the first
    /// try is the SOFI C5 residual. If that yields, the featomic hop
    /// is leftover `p − μ` on a defect or a kick of the mean SOAP
    /// when leftover is a closed-shell breath. Partitioned by
    /// `species`, restricted to `mobile`. `class` turns on the
    /// 555→421 / fcc prototype, cluster-only.
    Soap {
        /// Cartesian RMSD of the pulled-back step.
        rmsd: f64,
        /// Fixed SOAP cutoff in the same coordinate units as the state.
        cutoff: f64,
        /// Oracle residual (555 toward 421 / fcc). False is `2p − μ`.
        class: bool,
        /// Observed atomic numbers, one per point. Partitions the residual.
        species: Option<Vec<u32>>,
        /// Mobile atom indices. `None` is all atoms. Frozen stay as neighbours.
        mobile: Option<Vec<usize>>,
        /// Rigid groups. The ambient `J⁺` step is retracted to their
        /// nearest finite rigid motions before the proposal is returned.
        groups: Option<Vec<Vec<usize>>>,
    },
}

fn soap_arm(
    cfg: &Config,
    mobile: Option<Vec<usize>>,
    groups: Option<Vec<Vec<usize>>>,
) -> ClusterMove {
    let packing = cfg.species.is_none() && cfg.active_region.is_none() && cfg.frozen.is_none();
    // The Jacobian hop is a leftover step, a fraction of a bond, on
    // every system. Contact-scale amplitude was for a rigid group
    // translation; the pullback is not that move.
    let rmsd = LennardJonesPreset::SOAP_RMSD * cfg.length_scale;
    ClusterMove::Soap {
        rmsd,
        cutoff: LennardJonesPreset::SOAP_CUTOFF * cfg.length_scale,
        class: cfg.soap_class_residual && packing,
        species: cfg.species.clone(),
        mobile,
        groups: if cfg.soap_mode == SoapProposalMode::Rigid {
            groups
        } else {
            None
        },
    }
}

/// Pair energy per point, in the Lennard-Jones form Wales and Doye use to
/// decide which point is worst bound.
///
/// `E(i) = sum_{j != i} 4 [ (1/r_ij)^12 - (1/r_ij)^6 ]`, so the total energy is
/// half the sum. This compatibility entry point uses the reduced-unit preset.
pub fn pair_energies(x: ArrayView1<f64>, n: usize) -> Array1<f64> {
    pair_energies_scaled(
        x,
        n,
        LennardJonesPreset::REDUCED_SCALE,
        LennardJonesPreset::REDUCED_SCALE,
    )
}

/// Lennard-Jones pair energy per point against declared scales.
pub fn pair_energies_scaled(
    x: ArrayView1<f64>,
    n: usize,
    length_scale: f64,
    energy_scale: f64,
) -> Array1<f64> {
    let mut e = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 <= 0.0 {
                continue;
            }
            let scaled_r2 = length_scale * length_scale / r2;
            let s6 = scaled_r2 * scaled_r2 * scaled_r2;
            let v = 4.0 * energy_scale * (s6 * s6 - s6);
            e[i] += v;
            e[j] += v;
        }
    }
    e
}

/// Whether the worst-bound point is loose enough for an angular move, and which
/// one it is.
///
/// The criterion is the paper's: the highest pair energy rising above a
/// fraction `ratio` of the lowest. Both are negative for a bound cluster, so
/// this fires when the worst-bound point holds less than `ratio` of the binding
/// the best-bound one does.
pub fn worst_bound(x: ArrayView1<f64>, n: usize, ratio: f64) -> Option<usize> {
    worst_bound_scaled(
        x,
        n,
        ratio,
        LennardJonesPreset::REDUCED_SCALE,
        LennardJonesPreset::REDUCED_SCALE,
    )
}

/// Scale-aware form of [`worst_bound`].
pub fn worst_bound_scaled(
    x: ArrayView1<f64>,
    n: usize,
    ratio: f64,
    length_scale: f64,
    energy_scale: f64,
) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let e = pair_energies_scaled(x, n, length_scale, energy_scale);
    let mut hi = 0usize;
    let mut lo = 0usize;
    for i in 1..n {
        if e[i] > e[hi] {
            hi = i;
        }
        if e[i] < e[lo] {
            lo = i;
        }
    }
    if e[lo] >= 0.0 {
        return None;
    }
    if e[hi] > ratio * e[lo] {
        Some(hi)
    } else {
        None
    }
}

/// Rigidly relocates the least-bound group of a molecular cluster.
///
/// Contacts are counted between atoms of different groups only, so a tightly
/// bonded molecule does not read as well-bound by its own bonds. The chosen
/// group is translated so its centroid lands on a random direction at the
/// cluster's surface radius and rotated rigidly about its centroid by a
/// uniform random rotation, preserving intra-group geometry exactly.
pub(super) fn group_relocate<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    groups: &[Vec<usize>],
    neighbour_cutoff: f64,
    rng: &mut R,
) -> Array1<f64> {
    let mut out = x.to_owned();
    if groups.len() < 2 {
        return out;
    }
    let n = x.len() / 3;
    let mut owner = vec![usize::MAX; n];
    for (g, atoms) in groups.iter().enumerate() {
        for &a in atoms {
            if a < n {
                owner[a] = g;
            }
        }
    }
    // Inter-group contacts per group.
    let cut2 = neighbour_cutoff * neighbour_cutoff;
    let mut contacts = vec![0usize; groups.len()];
    for i in 0..n {
        for j in (i + 1)..n {
            if owner[i] == owner[j] || owner[i] == usize::MAX || owner[j] == usize::MAX {
                continue;
            }
            let d2: f64 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum();
            if d2 < cut2 {
                contacts[owner[i]] += 1;
                contacts[owner[j]] += 1;
            }
        }
    }
    let worst = (0..groups.len()).min_by_key(|&g| contacts[g]).unwrap_or(0);
    // Cluster centroid and surface radius from group centroids.
    let mut cc = [0.0_f64; 3];
    for i in 0..n {
        for k in 0..3 {
            cc[k] += x[3 * i + k];
        }
    }
    for v in cc.iter_mut() {
        *v /= n.max(1) as f64;
    }
    let mut rmax = 0.0_f64;
    for atoms in groups.iter() {
        let gc = group_centroid(x, atoms);
        let r: f64 = (0..3)
            .map(|k| (gc[k] - cc[k]) * (gc[k] - cc[k]))
            .sum::<f64>()
            .sqrt();
        rmax = rmax.max(r);
    }
    // Random direction on the sphere, random rigid rotation.
    let dir = {
        let mut v;
        loop {
            v = [
                rng.random::<f64>() * 2.0 - 1.0,
                rng.random::<f64>() * 2.0 - 1.0,
                rng.random::<f64>() * 2.0 - 1.0,
            ];
            let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if n2 > 1e-6 && n2 <= 1.0 {
                let nn = n2.sqrt();
                v = [v[0] / nn, v[1] / nn, v[2] / nn];
                break;
            }
        }
        v
    };
    let target = [
        cc[0] + dir[0] * rmax,
        cc[1] + dir[1] * rmax,
        cc[2] + dir[2] * rmax,
    ];
    let atoms = &groups[worst];
    let gc = group_centroid(x, atoms);
    // Uniform random rotation from three uniform angles is biased; a rotation
    // about a uniform random axis by a uniform angle is enough for a proposal
    // and keeps the code free of quaternion machinery.
    let axis = dir;
    let angle = rng.random::<f64>() * std::f64::consts::TAU;
    let (sa, ca) = angle.sin_cos();
    for &a in atoms {
        if a >= n {
            continue;
        }
        let p = [x[3 * a] - gc[0], x[3 * a + 1] - gc[1], x[3 * a + 2] - gc[2]];
        // Rodrigues rotation about `axis`.
        let dot = p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2];
        let cross = [
            axis[1] * p[2] - axis[2] * p[1],
            axis[2] * p[0] - axis[0] * p[2],
            axis[0] * p[1] - axis[1] * p[0],
        ];
        for k in 0..3 {
            let rot = p[k] * ca + cross[k] * sa + axis[k] * dot * (1.0 - ca);
            out[3 * a + k] = target[k] + rot;
        }
    }
    out
}

/// Connected components of the contact graph at `cutoff`: the molecules the
/// structure actually contains, read off its own bonding rather than declared.
///
/// The declared-group defect this replaces: a walker whose quench formed a new
/// covalent bond kept moving under groups that no longer matched its bonding,
/// so every rigid move proposed tearing a real bond, and the walker froze for
/// the rest of its run at the reacted species. Deriving the groups from the
/// current structure keeps the move library consistent with whatever chemistry
/// the surface has produced.
pub fn connectivity_groups(x: ArrayView1<f64>, n: usize, cutoff: f64) -> Vec<Vec<usize>> {
    let table = crate::neighbors::NeighborTable::build(x, n, cutoff);
    components(n, |a| table.neighbors(a).to_vec())
}

/// Which atoms are active around the seed set: the seeds themselves plus
/// everything within `shells` bond-matrix neighbour shells of them, computed
/// from the current coordinates.
///
/// The surface-search shape this serves: the adsorbate is the seed, the
/// substrate atoms it currently touches respond, and the far substrate stands
/// still, with the active region following the adsorbate as it moves. The
/// shell rule is the nearest-neighbour bound; a descriptor-deviation bound
/// over the same neighbourhoods is its refinement, not its replacement.
pub fn active_mask(
    x: ArrayView1<f64>,
    species: &[u32],
    seeds: &[usize],
    shells: usize,
    tolerance: f64,
) -> Vec<bool> {
    let n = species.len().min(x.len() / 3);
    let mut active = vec![false; n];
    let mut frontier: Vec<usize> = seeds.iter().copied().filter(|&a| a < n).collect();
    for &a in &frontier {
        active[a] = true;
    }
    for _ in 0..shells {
        let mut next = Vec::new();
        for &a in &frontier {
            let ra = covalent_radius(species[a]);
            for b in 0..n {
                if active[b] {
                    continue;
                }
                let cut = tolerance * (ra + covalent_radius(species[b]));
                if cut <= 0.0 {
                    continue;
                }
                let d2: f64 = (0..3)
                    .map(|k| {
                        let d = x[3 * a + k] - x[3 * b + k];
                        d * d
                    })
                    .sum();
                if d2.sqrt() < cut {
                    active[b] = true;
                    next.push(b);
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    active
}

/// Covalent radius in Angstrom by atomic number, Cordero and coworkers
/// (doi:10.1039/B801115J); zero for numbers outside the table, which makes an
/// unknown species bond to nothing rather than to everything.
pub fn covalent_radius(z: u32) -> f64 {
    // Single-bond covalent radii (A), Cordero et al., Dalton Trans. 2008,
    // 2832-2838 (doi:10.1039/B801115J). Index is Z; 0 is unused. Unknown
    // species stay 0 so they bond to nothing rather than to everything.
    const R: [f64; 97] = [
        0.0, // unused
        0.31, 0.28, // H He
        1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, // Li-Ne
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, // Na-Ar
        2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19,
        1.20, 1.20, 1.16, // K-Kr
        2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39,
        1.38, 1.39, 1.40, // Rb-Xe
        2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90,
        1.87, 1.87, // Cs-Lu
        1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50,
        1.50, // Hf-Rn
        2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, // Fr-Cm
    ];
    if (z as usize) < R.len() {
        R[z as usize]
    } else {
        0.0
    }
}

/// Connected components under the bond-matrix rule: two atoms bond when their
/// separation is below `tolerance` times the sum of their covalent radii, the
/// species-aware connectivity of the Berny and reaction-network lineage. A
/// single length cannot serve a system holding both hydrogen and copper; the
/// radii sums can.
pub fn connectivity_groups_z(
    x: ArrayView1<f64>,
    species: &[u32],
    tolerance: f64,
) -> Vec<Vec<usize>> {
    let n = species.len().min(x.len() / 3);
    components(n, |a| {
        let mut nb = Vec::new();
        let ra = covalent_radius(species[a]);
        for b in 0..n {
            if b == a {
                continue;
            }
            let cut = tolerance * (ra + covalent_radius(species[b]));
            if cut <= 0.0 {
                continue;
            }
            let d2: f64 = (0..3)
                .map(|k| {
                    let d = x[3 * a + k] - x[3 * b + k];
                    d * d
                })
                .sum();
            if d2.sqrt() < cut {
                nb.push(b);
            }
        }
        nb
    })
}

fn components(n: usize, neighbours: impl Fn(usize) -> Vec<usize>) -> Vec<Vec<usize>> {
    let mut seen = vec![false; n];
    let mut groups = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut comp = vec![start];
        seen[start] = true;
        let mut queue = vec![start];
        while let Some(a) = queue.pop() {
            for b in neighbours(a) {
                if !seen[b] {
                    seen[b] = true;
                    comp.push(b);
                    queue.push(b);
                }
            }
        }
        comp.sort_unstable();
        groups.push(comp);
    }
    groups
}

/// Centroid of one group.
fn group_centroid(x: ArrayView1<f64>, atoms: &[usize]) -> [f64; 3] {
    let n = x.len() / 3;
    let mut c = [0.0_f64; 3];
    let mut m = 0usize;
    for &a in atoms {
        if a < n {
            for k in 0..3 {
                c[k] += x[3 * a + k];
            }
            m += 1;
        }
    }
    for v in c.iter_mut() {
        *v /= m.max(1) as f64;
    }
    c
}

impl ClusterMove {
    /// The move library, configured for `n` points.
    ///
    /// The two plain perturbations come first and are not optional. Displacing
    /// every point uniformly is the move basin hopping is defined by, and the
    /// step of 0.38 is inside the 0.36 to 0.40 band Wales and Doye report for
    /// the quenched surface. A library of packing-changing moves alone leaves
    /// the chain with no way to make an ordinary small step, and measured on
    /// LJ38 at 400 thousand charged evaluations that library solved 1 seed in 8
    /// where the campaign driver, which carries both, solves 8.
    pub fn library(n: usize) -> Vec<ClusterMove> {
        Self::library_scaled(
            n,
            LennardJonesPreset::REDUCED_SCALE,
            LennardJonesPreset::NEIGHBOUR_CUTOFF,
            LennardJonesPreset::SYMMETRISE_CUTOFF,
        )
    }

    fn library_scaled(
        n: usize,
        length_scale: f64,
        neighbour_cutoff: f64,
        symmetrise_cutoff: f64,
    ) -> Vec<ClusterMove> {
        vec![
            ClusterMove::AllPoints {
                step: LennardJonesPreset::ALL_POINTS_STEP * length_scale,
            },
            ClusterMove::SinglePoint {
                n_points: n,
                step: LennardJonesPreset::SINGLE_POINT_STEP * length_scale,
            },
            ClusterMove::SurfaceRelocate(SurfaceRelocate {
                n_points: n,
                neighbour_cutoff,
            }),
            ClusterMove::ShellRotate(ShellRotate { n_points: n }),
            ClusterMove::Symmetrise(Symmetrise {
                n_points: n,
                orders: vec![2, 3, 4, 5, 6],
                pair_cutoff: symmetrise_cutoff,
            }),
        ]
    }

    /// The library without the arms measured to produce nothing.
    ///
    /// Traced over 72 runs at 38 points, every one of 55 funnel crossings and
    /// 617 of 634 ordinary improvements came from surface relocation, the
    /// single-point move and symmetrisation. The all-point isotropic move, the
    /// canonical basin-hopping perturbation, produced 8 improvements and no
    /// crossing from a fifth of the proposals; the shell rotation 9 and none.
    /// The realised crossing displacement has participation about 1/n: one
    /// atom carries it. Dropping the two inert arms reallocates two fifths of
    /// the proposal budget to the moves that do the work.
    /// The lean library with the burst arm added.
    pub fn library_lean_burst(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library_lean_scaled(
            n,
            LennardJonesPreset::REDUCED_SCALE,
            LennardJonesPreset::NEIGHBOUR_CUTOFF,
            LennardJonesPreset::SYMMETRISE_CUTOFF,
        );
        v.push(ClusterMove::Burst {
            n_points: n,
            neighbour_cutoff: LennardJonesPreset::NEIGHBOUR_CUTOFF,
        });
        v.push(ClusterMove::Soap {
            rmsd: LennardJonesPreset::SOAP_RMSD * LennardJonesPreset::REDUCED_SCALE,
            cutoff: LennardJonesPreset::SOAP_CUTOFF * LennardJonesPreset::REDUCED_SCALE,
            class: false,
            species: None,
            mobile: None,
            groups: None,
        });
        v
    }

    /// Builds the reduced-unit lean Lennard-Jones move library.
    pub fn library_lean(n: usize) -> Vec<ClusterMove> {
        Self::library_lean_scaled(
            n,
            LennardJonesPreset::REDUCED_SCALE,
            LennardJonesPreset::NEIGHBOUR_CUTOFF,
            LennardJonesPreset::SYMMETRISE_CUTOFF,
        )
    }

    fn library_lean_scaled(
        n: usize,
        length_scale: f64,
        neighbour_cutoff: f64,
        symmetrise_cutoff: f64,
    ) -> Vec<ClusterMove> {
        vec![
            ClusterMove::SinglePoint {
                n_points: n,
                step: LennardJonesPreset::SINGLE_POINT_STEP * length_scale,
            },
            ClusterMove::SurfaceRelocate(SurfaceRelocate {
                n_points: n,
                neighbour_cutoff,
            }),
            ClusterMove::Symmetrise(Symmetrise {
                n_points: n,
                orders: vec![2, 3, 4, 5, 6],
                pair_cutoff: symmetrise_cutoff,
            }),
        ]
    }

    /// The move library for a molecular cluster: every arm rigid on the
    /// caller's groups. Shake as the workhorse, relocation as the crossing
    /// operator, the composed burst as the excursion.
    pub fn library_molecular(groups: Vec<Vec<usize>>, neighbour_cutoff: f64) -> Vec<ClusterMove> {
        Self::library_molecular_scaled(groups, neighbour_cutoff, LennardJonesPreset::REDUCED_SCALE)
    }

    fn library_molecular_scaled(
        groups: Vec<Vec<usize>>,
        neighbour_cutoff: f64,
        length_scale: f64,
    ) -> Vec<ClusterMove> {
        vec![
            ClusterMove::GroupShake {
                groups: groups.clone(),
                amplitude: LennardJonesPreset::GROUP_SHAKE * length_scale,
            },
            ClusterMove::GroupRelocate {
                groups: groups.clone(),
                neighbour_cutoff,
            },
            ClusterMove::GroupBurst {
                groups,
                neighbour_cutoff,
            },
        ]
    }

    /// The combined reactive library: the rigid-group arms carry molecular
    /// transport, the atomic arms keep bond breaking and forming reachable.
    /// The allocator owns the split, so a rigid system starves the atomic
    /// arms and a reactive event revives them. The atomic arms share the
    /// inter-group cutoff: on a molecular system the surface is defined by
    /// contacts at that scale, not at the bonded scale.
    pub fn library_combined(
        n: usize,
        groups: Vec<Vec<usize>>,
        neighbour_cutoff: f64,
    ) -> Vec<ClusterMove> {
        Self::library_combined_scaled(
            n,
            groups,
            neighbour_cutoff,
            LennardJonesPreset::REDUCED_SCALE,
        )
    }

    fn library_combined_scaled(
        n: usize,
        groups: Vec<Vec<usize>>,
        neighbour_cutoff: f64,
        length_scale: f64,
    ) -> Vec<ClusterMove> {
        let mut v = Self::library_molecular_scaled(groups, neighbour_cutoff, length_scale);
        v.push(ClusterMove::SinglePoint {
            n_points: n,
            step: LennardJonesPreset::SINGLE_POINT_STEP * length_scale,
        });
        v.push(ClusterMove::SurfaceRelocate(SurfaceRelocate {
            n_points: n,
            neighbour_cutoff,
        }));
        v.push(ClusterMove::Burst {
            n_points: n,
            neighbour_cutoff,
        });
        v
    }

    /// The library with the heavy-tailed visiting move added.
    pub fn library_with_visit(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library(n);
        v.push(ClusterMove::Visit { q_v: 2.7 });
        v
    }

    /// The library with the twin move added.
    ///
    /// Separate from the reseeding library because the two are different bets.
    /// A reseed discards the structure; a twin keeps all of it but one
    /// boundary layer, which is the whole reason to expect it to be accepted.
    pub fn library_with_twin(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library(n);
        v.push(ClusterMove::Twin { n_points: n });
        v
    }

    /// The library with the reseeding moves added.
    ///
    /// Separate because these are the only proposals that discard the current
    /// structure rather than perturb it, so a caller comparing against the
    /// displacement-only search needs to be able to ask for one or the other.
    /// Every source is offered; which is worth drawing is the allocator's
    /// question, not this function's.
    pub fn library_with_reseed(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library(n);
        for source in crate::lattice::Source::library() {
            v.push(ClusterMove::Reseed {
                n_points: n,
                source,
            });
        }
        v
    }

    /// The library with one growth arm that regrows the structure from the
    /// local order it already has.
    ///
    /// The full reseeding library carries four named packings alongside this
    /// one, and on a system whose global minimum is one of those packings the
    /// move can build the answer rather than search for it: at 38 points the
    /// full library solves 72 of 72, and on 75 and 98 points, whose
    /// morphologies it does not carry, it is worse than no reseeding at all.
    ///
    /// This arm carries no packing. [`crate::lattice::Source::Observed`] reads
    /// the local order out of the structure the chain is standing on and
    /// regrows from that, so nothing external enters and no answer can be
    /// encoded. It isolates whether the effect belongs to rebuilding
    /// coherently or to holding the right template.
    pub fn library_with_self_reseed(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library(n);
        v.push(ClusterMove::Reseed {
            n_points: n,
            source: crate::lattice::Source::Observed,
        });
        v
    }

    /// Growth arms and a twin arm together, with the source chosen by a
    /// posterior.
    ///
    /// Growth from a local order reaches a morphology whose defining order sits
    /// in the first coordination shell, which is what face-centred cubic and
    /// icosahedral packings are. It cannot reach one defined by a global
    /// arrangement: a Marks decahedron is face-centred cubic tetrahedra joined
    /// across twin planes, locally close packed everywhere, with the five-fold
    /// character living in the joins. No local template expresses that, and
    /// measurement says so -- every growth-only arm scores far above the
    /// control at 38 points, whose answer is a face-centred cubic truncated
    /// octahedron, and below it at 75 points, whose answer is decahedral.
    ///
    /// The twin move supplies the missing kind of order. The two together span
    /// the morphologies clusters actually adopt, and the posterior decides
    /// which the system in front of it wants rather than the caller.
    pub fn library_with_growth_and_twin(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library_with_learned_reseed(n);
        v.push(ClusterMove::Twin { n_points: n });
        v
    }

    /// The library with a single growth arm, for when a posterior picks the
    /// construction.
    ///
    /// Five arms are five arms only while the arm decides what is built. Once
    /// the constructor chooses the source itself, the labels no longer describe
    /// the proposals, and the allocator is estimating five accept rates for one
    /// move from evidence that has been shuffled between them.
    pub fn library_with_learned_reseed(n: usize) -> Vec<ClusterMove> {
        let mut v = Self::library(n);
        v.push(ClusterMove::Reseed {
            n_points: n,
            source: crate::lattice::Source::Observed,
        });
        v
    }

    /// Short name, for per-arm reporting.
    pub fn name(&self) -> String {
        match self {
            ClusterMove::AllPoints { .. } => "all".into(),
            ClusterMove::SinglePoint { .. } => "single".into(),
            ClusterMove::SurfaceRelocate(_) => "surface".into(),
            ClusterMove::GroupRelocate { .. } => "gsurface".into(),
            ClusterMove::GroupShake { .. } => "gshake".into(),
            ClusterMove::GroupBurst { .. } => "gburst".into(),
            ClusterMove::Burst { .. } => "burst".into(),
            ClusterMove::ShellRotate(_) => "shell".into(),
            ClusterMove::Symmetrise(_) => "sym".into(),
            ClusterMove::Visit { .. } => "visit".into(),
            ClusterMove::Angular { .. } => "angular".into(),
            ClusterMove::Twin { .. } => "twin".into(),
            ClusterMove::Soap { .. } => "soap".into(),
            ClusterMove::Reseed { source, .. } => format!("grow:{}", source.name()),
        }
    }

    /// Draws a proposal from whichever kernel this is.
    pub fn propose<R: Rng + ?Sized>(&self, x: ArrayView1<f64>, t: f64, rng: &mut R) -> Array1<f64> {
        self.propose_scaled(x, t, 1.0, rng)
    }

    /// Draws a proposal with the amplitude multiplied by `scale`.
    ///
    /// Only the two plain perturbations carry an amplitude to scale. Surface
    /// relocation, shell rotation and symmetrisation change a packing rather
    /// than displace by a length, so there is nothing for a scale to multiply
    /// and they are drawn unchanged.
    pub fn propose_scaled<R: Rng + ?Sized>(
        &self,
        x: ArrayView1<f64>,
        t: f64,
        scale: f64,
        rng: &mut R,
    ) -> Array1<f64> {
        match self {
            ClusterMove::AllPoints { step } => {
                let mut y = x.to_owned();
                let h = step * scale;
                for v in y.iter_mut() {
                    *v += rng.random_range(-h..h);
                }
                y
            }
            ClusterMove::SinglePoint { n_points, step } => {
                let mut y = x.to_owned();
                let i = rng.random_range(0..*n_points);
                let h = step * scale;
                for k in 0..3 {
                    y[3 * i + k] += rng.random_range(-h..h);
                }
                y
            }
            ClusterMove::Twin { n_points } => crate::twin::propose(x, *n_points, rng),
            ClusterMove::Reseed { n_points, source } => {
                // Both the order and the length scale are read off the current
                // structure, so the move carries no knowledge of the potential
                // and none of the objective it is proposing against.
                crate::lattice::candidate(*source, x, *n_points, rng)
            }
            ClusterMove::Angular {
                n_points,
                length_scale,
                energy_scale,
            } => {
                let n = *n_points;
                let mut y = x.to_owned();
                if n == 0 {
                    return y;
                }
                // The centre of mass, and the radius of the point furthest from
                // it, which is where the moved point lands.
                let mut c = [0.0_f64; 3];
                for i in 0..n {
                    for k in 0..3 {
                        c[k] += y[3 * i + k];
                    }
                }
                for v in c.iter_mut() {
                    *v /= n as f64;
                }
                let mut rmax = 0.0_f64;
                for i in 0..n {
                    let dx = y[3 * i] - c[0];
                    let dy = y[3 * i + 1] - c[1];
                    let dz = y[3 * i + 2] - c[2];
                    rmax = rmax.max((dx * dx + dy * dy + dz * dz).sqrt());
                }
                let i = worst_bound_scaled(y.view(), n, 0.42, *length_scale, *energy_scale)
                    .unwrap_or_else(|| {
                        let e = pair_energies_scaled(y.view(), n, *length_scale, *energy_scale);
                        let mut hi = 0usize;
                        for k in 1..n {
                            if e[k] > e[hi] {
                                hi = k;
                            }
                        }
                        hi
                    });
                // Uniform on the sphere: cos(theta) uniform in [-1, 1], not
                // theta itself, or the poles are oversampled.
                let cos_t: f64 = rng.random_range(-1.0..1.0);
                let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
                let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                y[3 * i] = c[0] + rmax * sin_t * phi.cos();
                y[3 * i + 1] = c[1] + rmax * sin_t * phi.sin();
                y[3 * i + 2] = c[2] + rmax * cos_t;
                y
            }
            ClusterMove::Visit { q_v } => {
                crate::movekernel::TsallisVisit::new(*q_v).propose(x, t, rng)
            }
            ClusterMove::GroupRelocate {
                groups,
                neighbour_cutoff,
            } => group_relocate(x, groups, *neighbour_cutoff, rng),
            ClusterMove::GroupShake { groups, amplitude } => {
                let mut out = x.to_owned();
                if groups.is_empty() {
                    return out;
                }
                let g = rng.random_range(0..groups.len());
                let atoms = &groups[g];
                let gc = group_centroid(x, atoms);
                let mut shift = [0.0_f64; 3];
                for v in shift.iter_mut() {
                    let u1: f64 = rng.random::<f64>().max(1e-12);
                    let u2: f64 = rng.random::<f64>();
                    *v = amplitude * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                }
                let axis = {
                    let mut v;
                    loop {
                        v = [
                            rng.random::<f64>() * 2.0 - 1.0,
                            rng.random::<f64>() * 2.0 - 1.0,
                            rng.random::<f64>() * 2.0 - 1.0,
                        ];
                        let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                        if n2 > 1e-6 && n2 <= 1.0 {
                            let nn = n2.sqrt();
                            v = [v[0] / nn, v[1] / nn, v[2] / nn];
                            break;
                        }
                    }
                    v
                };
                // A modest rotation, half a radian at most, so the shake stays
                // a shake rather than a relocation.
                let angle = rng.random::<f64>() - 0.5;
                let (sa, ca) = angle.sin_cos();
                let n = x.len() / 3;
                for &a in atoms {
                    if a >= n {
                        continue;
                    }
                    let pvec = [x[3 * a] - gc[0], x[3 * a + 1] - gc[1], x[3 * a + 2] - gc[2]];
                    let dot = pvec[0] * axis[0] + pvec[1] * axis[1] + pvec[2] * axis[2];
                    let cross = [
                        axis[1] * pvec[2] - axis[2] * pvec[1],
                        axis[2] * pvec[0] - axis[0] * pvec[2],
                        axis[0] * pvec[1] - axis[1] * pvec[0],
                    ];
                    for k in 0..3 {
                        let rot = pvec[k] * ca + cross[k] * sa + axis[k] * dot * (1.0 - ca);
                        out[3 * a + k] = gc[k] + shift[k] + rot;
                    }
                }
                out
            }
            ClusterMove::GroupBurst {
                groups,
                neighbour_cutoff,
            } => {
                let mut cur = group_relocate(x, groups, *neighbour_cutoff, rng);
                let mut hops = 1;
                while hops < 8 && rng.random::<f64>() < 2.0 / 3.0 {
                    cur = group_relocate(cur.view(), groups, *neighbour_cutoff, rng);
                    hops += 1;
                }
                cur
            }
            ClusterMove::Burst {
                n_points,
                neighbour_cutoff,
            } => {
                let kernel = SurfaceRelocate {
                    n_points: *n_points,
                    neighbour_cutoff: *neighbour_cutoff,
                };
                let mut cur = kernel.propose(x, t, rng);
                // Geometric continuation at 2/3: mean three relocations,
                // capped so a long tail cannot spend a whole structure.
                let mut hops = 1;
                while hops < 8 && rng.random::<f64>() < 2.0 / 3.0 {
                    cur = kernel.propose(cur.view(), t, rng);
                    hops += 1;
                }
                cur
            }
            ClusterMove::SurfaceRelocate(k) => k.propose(x, t, rng),
            ClusterMove::ShellRotate(k) => k.propose(x, t, rng),
            ClusterMove::Symmetrise(k) => k.propose(x, t, rng),
            ClusterMove::Soap {
                rmsd,
                cutoff,
                class,
                species,
                mobile,
                groups,
            } => {
                let spec = crate::soap::SoapSpec {
                    n_max: 3,
                    l_max: 6,
                    rcut_nn: *cutoff,
                };
                let packing = species.is_none() && mobile.is_none();
                if *class && packing {
                    crate::soap::step_away(x, &[], spec, *rmsd, rng)
                } else {
                    crate::soap::step_away_cloud(
                        x,
                        spec,
                        *rmsd,
                        species.as_deref(),
                        mobile.as_deref(),
                        groups.as_deref(),
                        rng,
                    )
                }
            }
        }
    }
}

impl MoveKernel<f64> for ClusterMove {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, t: f64, rng: &mut R) -> Array1<f64> {
        ClusterMove::propose(self, i, t, rng)
    }

    fn supports_in<N: crate::neigh::Neighborhood<f64>>(&self, _n: &N) -> bool {
        true
    }
}

/// Uniform mixture of the proposal kernels selected by a cluster preset.
pub struct ClusterProposal {
    kernels: Vec<ClusterMove>,
}

impl ClusterProposal {
    pub(super) fn new(kernels: Vec<ClusterMove>) -> Self {
        assert!(!kernels.is_empty(), "a proposal library must not be empty");
        Self { kernels }
    }

    /// Number of proposal arms in the mixture.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the mixture has no proposal arms.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

impl MoveKernel<f64> for ClusterProposal {
    fn propose<R: Rng + ?Sized>(&self, i: ArrayView1<f64>, t: f64, rng: &mut R) -> Array1<f64> {
        let arm = rng.random_range(0..self.kernels.len());
        self.kernels[arm].propose(i, t, rng)
    }

    fn supports_in<N: crate::neigh::Neighborhood<f64>>(&self, _n: &N) -> bool {
        true
    }
}

#[cfg(test)]
mod move_scaling_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// The escape scale has to reach the step, which is the whole mechanism.
    /// It did not: the amplitude moves ignored the temperature argument, so a
    /// controller multiplying it changed nothing about how far a proposal
    /// reached.
    #[test]
    fn scaling_widens_the_displacement() {
        let mv = ClusterMove::AllPoints { step: 0.4 };
        let x: Array1<f64> = Array1::zeros(30);
        let spread = |scale: f64| {
            let mut rng = StdRng::seed_from_u64(7);
            let mut worst = 0.0_f64;
            for _ in 0..200 {
                let y = mv.propose_scaled(x.view(), 0.8, scale, &mut rng);
                for v in y.iter() {
                    worst = worst.max(v.abs());
                }
            }
            worst
        };
        let one = spread(1.0);
        let four = spread(4.0);
        assert!(
            (four / one - 4.0).abs() < 0.2,
            "a scale of four should reach four times as far: {four} against {one}"
        );
    }

    #[test]
    fn soap_pullback_proposal_moves_more_than_one_atom() {
        let mv = ClusterMove::Soap {
            rmsd: 0.45,
            cutoff: 3.5,
            class: true,
            species: None,
            mobile: None,
            groups: None,
        };
        let x = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.15, 0.08, 0.02, 0.18, 1.22, 0.11, 0.95, 0.85, 1.28,
        ]);
        let mut rng = StdRng::seed_from_u64(4);
        let y = mv.propose(x.view(), 0.8, &mut rng);
        let n = 4;
        let mut moved = 0usize;
        for i in 0..n {
            let mut d2 = 0.0;
            for k in 0..3 {
                let d = y[3 * i + k] - x[3 * i + k];
                d2 += d * d;
            }
            if d2.sqrt() > 0.05 {
                moved += 1;
            }
        }
        assert!(moved >= 2, "SOAP proposal moved {moved} atoms");
    }

    #[test]
    fn recommended_soap_uses_lmax6() {
        let rec = Config::recommended(13);
        let soap = rec
            .move_library
            .kernels(&rec)
            .into_iter()
            .find(|k| matches!(k, ClusterMove::Soap { .. }));
        match soap {
            Some(ClusterMove::Soap { cutoff, .. }) => {
                let spec = crate::soap::SoapSpec {
                    n_max: 3,
                    l_max: 6,
                    rcut_nn: cutoff,
                };
                assert_eq!(spec.l_max, 6);
                assert!(crate::ace::dim(3, 6) > crate::ace::dim(3, 3));
            }
            _ => panic!("recommended LeanBurst has no SOAP arm"),
        }
    }

    #[test]
    fn recommended_leanburst_offers_soap() {
        let rec = Config::recommended(13);
        let names: Vec<String> = rec
            .move_library
            .kernels(&rec)
            .iter()
            .map(|k| k.name())
            .collect();
        assert!(
            names.iter().any(|n| n == "soap"),
            "recommended LeanBurst missing soap: {names:?}"
        );
        assert!(
            !rec.soap_class_residual,
            "recommended SOAP must not ship the 555→421 / fcc oracle"
        );
        assert!(
            rec.move_library
                .kernels(&rec)
                .iter()
                .any(|k| matches!(k, ClusterMove::Soap { class: false, .. })),
            "recommended SOAP arm is not the observed-cloud residual"
        );
        assert!(
            rec.adaptive_height,
            "recommended must fill a packing over many revisits"
        );
        assert!(
            (rec.height_revisits - 20.0).abs() < 1e-12,
            "recommended N_f analogue is 20 revisits, got {}",
            rec.height_revisits
        );
        #[cfg(feature = "featomic")]
        {
            assert_eq!(rec.keying, Keying::SoapPacking);
            assert!(
                (rec.merge_radius - crate::featomic_hop::SOAP_PACK_MERGE).abs() < 1e-12,
                "SOAP packing merge is {}, got {}",
                crate::featomic_hop::SOAP_PACK_MERGE,
                rec.merge_radius
            );
        }
    }

    #[test]
    fn recommended_molecular_offers_soap_with_species() {
        let rec = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
        assert_eq!(rec.soap_mode, SoapProposalMode::Flexible);
        assert!(
            !rec.packing_cna_applies(),
            "CNA 555 must not apply to a molecule"
        );
        assert!(!rec.symmetrise_on_stall);
        assert!(
            rec.move_library.kernels(&rec).iter().any(|k| matches!(
                k,
                ClusterMove::Soap {
                    class: false,
                    species: Some(_),
                    groups: Some(_),
                    ..
                }
            )),
            "recommended_molecular missing species-aware SOAP: {:?}",
            rec.move_library
                .kernels(&rec)
                .iter()
                .map(|k| k.name())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn molecular_soap_modes_control_internal_deformation() {
        // The fivefold cloud has an ACE residual above the proposal floor.
        // One declared group isolates ambient deformation from its Kabsch
        // retraction without changing the descriptor or its activation gate.
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let vertices = [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ];
        let scale = 2.0_f64.powf(1.0 / 6.0) / (1.0 + phi * phi).sqrt();
        let mut x = Array1::zeros(39);
        for (index, vertex) in vertices.iter().enumerate() {
            for axis in 0..3 {
                x[3 * (index + 1) + axis] = scale * vertex[axis];
            }
        }
        let species = vec![1; 13];
        let groups = vec![(0..13).collect::<Vec<_>>()];
        let internal = |y: &Array1<f64>| {
            groups
                .iter()
                .flat_map(|group| {
                    (0..group.len()).flat_map(move |a| {
                        ((a + 1)..group.len()).map(move |b| {
                            let i = group[a];
                            let j = group[b];
                            (0..3)
                                .map(|axis| {
                                    let d = y[3 * i + axis] - y[3 * j + axis];
                                    d * d
                                })
                                .sum::<f64>()
                                .sqrt()
                        })
                    })
                })
                .collect::<Vec<_>>()
        };

        let mut flexible = Config::recommended_molecular(species.clone(), groups.clone(), 1.0);
        flexible.length_scale = 1.0;
        flexible.soap_mode = SoapProposalMode::Flexible;
        let flexible_move = flexible
            .move_library
            .kernels(&flexible)
            .into_iter()
            .find(|kernel| matches!(kernel, ClusterMove::Soap { .. }))
            .expect("flexible mode has no SOAP proposal");

        let mut rigid = flexible.clone();
        rigid.soap_mode = SoapProposalMode::Rigid;
        let rigid_move = rigid
            .move_library
            .kernels(&rigid)
            .into_iter()
            .find(|kernel| matches!(kernel, ClusterMove::Soap { .. }))
            .expect("rigid mode has no SOAP proposal");

        let mut rng_flexible = StdRng::seed_from_u64(17);
        let mut rng_rigid = StdRng::seed_from_u64(17);
        let y_flexible = flexible_move.propose(x.view(), 0.8, &mut rng_flexible);
        let y_rigid = rigid_move.propose(x.view(), 0.8, &mut rng_rigid);
        let d0 = internal(&x);
        let df = internal(&y_flexible);
        let dr = internal(&y_rigid);

        assert!(
            df.iter()
                .zip(&d0)
                .any(|(after, before)| (after - before).abs() > 1e-8),
            "flexible SOAP did not deform any internal distance"
        );
        for (after, before) in dr.iter().zip(&d0) {
            assert!(
                (after - before).abs() < 1e-10,
                "rigid SOAP changed {before} to {after}"
            );
        }

        let mut off = flexible;
        off.soap_mode = SoapProposalMode::Off;
        assert!(
            off.move_library
                .kernels(&off)
                .iter()
                .all(|kernel| !matches!(kernel, ClusterMove::Soap { .. })),
            "off mode retained a SOAP proposal"
        );
    }

    #[test]
    fn slab_soap_arm_carries_the_mobile_mask() {
        let mut rec = Config::recommended_molecular(vec![29, 29, 1, 1], vec![vec![2, 3]], 1.0);
        rec.active_region = Some((vec![2, 3], 0));
        rec.n_points = 4;
        let soap = rec
            .move_library
            .kernels(&rec)
            .into_iter()
            .find(|k| matches!(k, ClusterMove::Soap { .. }));
        match soap {
            Some(ClusterMove::Soap {
                class,
                species,
                mobile,
                ..
            }) => {
                assert!(!class, "slab SOAP must not be the 555->421 oracle");
                assert_eq!(species.as_deref(), Some(&[29, 29, 1, 1][..]));
                assert_eq!(mobile.as_deref(), Some(&[2, 3][..]));
            }
            _ => panic!("slab recommended library has no SOAP arm"),
        }
    }

    #[test]
    fn the_unscaled_call_is_the_old_behaviour() {
        let mv = ClusterMove::SinglePoint {
            n_points: 10,
            step: 1.0,
        };
        let x: Array1<f64> = Array1::zeros(30);
        let mut a = StdRng::seed_from_u64(3);
        let mut b = StdRng::seed_from_u64(3);
        let p = mv.propose(x.view(), 0.8, &mut a);
        let q = mv.propose_scaled(x.view(), 0.8, 1.0, &mut b);
        assert_eq!(p, q);
    }
}
