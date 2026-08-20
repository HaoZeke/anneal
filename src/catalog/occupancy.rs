//! Occupancy leave. Not Hyperband successive-halving.
//!
//! ## Packing identity (DECAF)
//!
//! Per-center SOAP class histograms. Same family iff
//! \(\|h-h'\|_1 \le\) [`crate::catalog::PACKING_MERGE`] \(= 0.20\).
//! Sealed LJ75 icosahedral-versus-Marks L1 is \(0.69\).
//!
//! ## Inverted Gelman--Rubin
//!
//! On the family-label series of assigned walks,
//! \(\hat R <\) [`crate::catalog::MIXED_RHAT`] \(= 1.2\) with
//! \(n_{\mathrm{assigned}} \ge 2\) is collapse: extras Leave. Distinct
//! packing labels are unmixed and do not force Leave.
//!
//! ## Ranking
//!
//! Champion of a family: lowest energy. Keep
//! \(\lfloor n_{\mathrm{extra}} / \eta \rfloor\) extras,
//! \(\eta =\) [`crate::catalog::REDUCTION_FACTOR`] \(= 3\). Surplus
//! Leave. Rank as soon as DECAF assigns a family. This is a keep
//! fraction, not a Li--Jamieson resource schedule.
//!
//! ## Superbasin
//!
//! Chatterjee--Voter AS-KMC: an intra-well hop with bias height
//! \(v_i \ge N_f w_0\) is frequent. Refuse the accept with probability
//! \(1/2\) (\(\alpha = 2\)). That is the well exit. A Feynman--Kac
//! epoch later resamples with the same ranking; it is not the exit.
//!
//! ## Leave start
//!
//! Another DECAF family already on file and packing not saturated:
//! take a catalog representative of the least-occupied one (funnel
//! exchange). After packing saturation, or if no other family is on
//! file: record the occupied packing in the shared SOAP archive and
//! step into a hole of that archive, or amplify the fivefold residual
//! when the archive is empty. The hole and packing-kick Cartesian
//! step is ring-lensed: pentagon atoms move when the occupied
//! profile has pentagons, triangle atoms when it does not. Champion
//! leftover SOAP is not that lens. Occupancy extras do not draw a
//! random cluster.
//! A single hole-and-quench returns to the same family; the hop
//! then requenches and widens until DECAF says the family changed.
//!
//! ## Modes
//!
//! Serial recommended: independent seeds, each tries to find the
//! published GM. Hit rate (58/72 Oh) is that mode.
//! Occupancy: the ensemble divides the PES. Each DECAF family has one
//! champion (lowest energy) that walks isomers. Extras of that family
//! Leave. Success is find-and-certify the putative GM, not every
//! replica landing on it.
//!
//! ## Stop
//!
//! A mixing certificate names a putative: uniquely deepest, occupant
//! mixed. A mixed competitor is required only when a competitor is
//! on file. A lone mixed floor is Gelman--Rubin on the sampled mode.
//! Unseen funnels are leftover-dwell and EI, not a second R-hat.
//! Packing Good--Turing names completeness of the seen codebook
//! (`packing_saturated`). Leftover-SOAP arrivals stay the hole
//! generator; leftover Good--Turing is not the stop. Hop re-observes
//! of the same well are not draws. A saturated packing census of
//! shallow families is not retire: extras keep Leaving so an unseen
//! funnel can still appear. Replicas retire when a mixing putative is
//! certified and that packing census is saturated and leftover SOAP
//! has dwelt under the unseen-mass ceiling (or the packing book holds
//! one family) and FunnelModel EI on the seen packings is exhausted
//! and the rematched family floor is met. The floor is the Fiedler split of the hop
//! graph after DECAF labels the sides: two when the seam separates
//! distinct packing families, one when the seam is leftover wells of
//! one packing (a superbasin). A landfold (Torgerson MDS of DECAF L1,
//! then 2-means) floor and a Franzblau primitive-ring floor are
//! reported beside it and do not retire.
//! `CATALOG_MIN_FAMILIES` is an override.
//! Occupant \(\hat R\) uses
//! [`crate::catalog::CERTIFY_MIN_SAMPLES`] traces; two-point quenches
//! on two random-start families are not a certificate.
//! A published energy (Cambridge or otherwise) is a score, not a
//! stop. Leftover-SOAP saturation on one family is collapse, not
//! completeness.

/// Actions that must land off the occupied leftover-SOAP well.
pub fn is_occupancy_leave_action(action: &str) -> bool {
    matches!(
        action,
        "hyperband_reseed" | "catalog_leave" | "population_reseed"
    )
}

/// How an occupancy Leave updates the live chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLeaveAdopt {
    /// Take the quenched trial as the live minimum.
    Quench,
    /// The quench sat in the occupied family. Keep the unquenched hole step.
    HoleStep,
    /// Same-family leftover hole. Do not adopt; the extra stays put
    /// and the hop loop draws another Leave.
    Refuse,
}

/// Where an occupancy extra goes when it Leaves.
///
/// Funnel exchange first, only while packing is unsaturated: a
/// catalog representative of a different, under-occupied packing.
/// That is how Leave includes Oh once Oh is on file. After packing
/// saturation, or if no other family is on file, a hole of the
/// shared occupied-packing archive, not a random cluster. Serial
/// recommended is a different mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLeaveTarget {
    /// Coordinator has a representative of a different DECAF family.
    OtherFamily,
    /// Shared SOAP archive hole, or fivefold residual if the archive is empty.
    ArchiveHole,
}

/// Consecutive leftover-SOAP occupancy_gt records under the unseen-mass
/// ceiling that count as a dwell. One leftover-sat nick is not enough:
/// a hatch raises \(\hat p_0\) by \((n-n_1)/(n(n+1))\).
pub const LEFTOVER_SAT_DWELL: usize = 5;

/// Bank and CSA turn FunnelModel EI on at three observed morphologies.
pub const OCCUPANCY_EI_MIN_OBS: usize = 3;

/// Jones remaining improvement on seen packing morphologies.
///
/// The FunnelModel is the bank/CSA GP. Exhausted when that model has
/// the bank observation floor and the largest EI at observed packings
/// is at most the model's noise. Unseen families are leftover-dwell,
/// not a far-field GP probe.
pub fn occupancy_ei_exhausted(max_ei: f64, n_obs: usize, noise: f64) -> bool {
    n_obs >= OCCUPANCY_EI_MIN_OBS && max_ei.is_finite() && max_ei <= noise
}

/// Leftover dwell from consecutive leftover-sat bits, newest last.
///
/// The last [`LEFTOVER_SAT_DWELL`] records must all be saturated.
/// A one-shot nick, or a nick then a hatch, is not a dwell.
pub fn leftover_sat_dwell(consecutive: &[bool]) -> bool {
    consecutive.len() >= LEFTOVER_SAT_DWELL
        && consecutive
            .iter()
            .rev()
            .take(LEFTOVER_SAT_DWELL)
            .all(|&sat| sat)
}

/// Leave destination. After packing saturation OtherFamily only
/// rematches families on file, so Leave is the archive hole.
pub fn occupancy_leave_target(
    other_family_in_catalog: bool,
    packing_saturated: bool,
) -> OccupancyLeaveTarget {
    if packing_saturated || !other_family_in_catalog {
        OccupancyLeaveTarget::ArchiveHole
    } else {
        OccupancyLeaveTarget::OtherFamily
    }
}

/// Occupancy Leave that quenches onto a new DECAF family is taken.
/// A leftover-SOAP hole that stays in the occupied family is refused:
/// off-well ico is still ico. Reseeds take the new start even when
/// DECAF still names the old family.
pub fn occupancy_leave_adopt(action: &str, family_changed: bool) -> Option<OccupancyLeaveAdopt> {
    if !is_occupancy_leave_action(action) {
        return None;
    }
    if action == "catalog_leave" && !family_changed {
        Some(OccupancyLeaveAdopt::Refuse)
    } else {
        Some(OccupancyLeaveAdopt::Quench)
    }
}

/// Why occupancy may retire a replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyCertificate {
    /// Occupant chains mixed onto a uniquely deepest attractor.
    /// A competitor, when one exists, must be mixed and less occupied.
    MixingCertified,
    /// Packing Good--Turing unseen mass is small, and the rematched
    /// family count meets the measured Fiedler-and-DECAF floor.
    CatalogSaturated,
}

impl OccupancyCertificate {
    /// Stable token for worker logs (`mixing` or `saturated`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MixingCertified => "mixing",
            Self::CatalogSaturated => "saturated",
        }
    }
}

/// Published energy match. A score for known hurdles, never a stop.
pub fn published_energy_score(best: f64, published: Option<f64>) -> bool {
    published.is_some_and(|target| best < target + 1e-4)
}

/// Default floor on rematched occupied families. `1` is Good--Turing
/// alone: stop when no new packings appear.
pub const DEFAULT_MIN_OCCUPIED_FAMILIES: usize = 1;

/// Same seam as `commission_bridge`: conductance below this is two
/// weakly coupled communities of the explored landscape.
pub const OCCUPANCY_SEAM_CONDUCTANCE: f64 = 0.1;

/// Family floor from the landscape Fiedler split after DECAF labels
/// the sides.
///
/// The hop-graph Fiedler vector names two weakly coupled leftover-SOAP
/// communities. Superbasin merge is leftover wells of one packing, so
/// a seam whose basins are the same DECAF family is one community.
/// Two is only when the seam separates distinct rematched packings.
/// A one-sided or well-mixed graph is one community.
/// `CATALOG_MIN_FAMILIES` remains an override.
pub fn occupancy_family_floor(
    conductance: Option<f64>,
    n_left: usize,
    n_right: usize,
    distinct_packing_sides: bool,
) -> usize {
    match conductance {
        Some(c)
            if c < OCCUPANCY_SEAM_CONDUCTANCE
                && n_left > 0
                && n_right > 0
                && distinct_packing_sides =>
        {
            2
        }
        _ => DEFAULT_MIN_OCCUPIED_FAMILIES,
    }
}

/// Secondary family floor from a 2-D folding of packing histograms.
///
/// Torgerson MDS of DECAF L1, then a 2-means split. Two when the sides
/// rematch to distinct packings and the centroids do not overlap.
/// Live rematch is the wrong input: fold the packing book so a family
/// extras have Left still sits on the map. A 2-means split is a
/// bipartition, not a leftover-cloud count; FES maxima stay the
/// landfold figure path. This is not the hop-graph Fiedler split and
/// it does not retire.
pub fn occupancy_map_floor(xy: &[[f64; 2]], family: &[usize]) -> usize {
    occupancy_map_split(xy, family).0
}

/// Map floor and the 2-means side counts.
pub fn occupancy_map_split(xy: &[[f64; 2]], family: &[usize]) -> (usize, usize, usize) {
    if xy.len() < 2 || xy.len() != family.len() {
        return (1, xy.len(), 0);
    }
    let Some((left, right)) = two_means(xy) else {
        return (1, xy.len(), 0);
    };
    if left.is_empty() || right.is_empty() {
        return (1, xy.len(), 0);
    }
    if family.iter().all(|&f| f == family[0]) {
        return (1, left.len(), right.len());
    }
    let Some(left_family) = majority_family(family, &left) else {
        return (1, left.len(), right.len());
    };
    let Some(right_family) = majority_family(family, &right) else {
        return (1, left.len(), right.len());
    };
    if left_family == right_family {
        return (1, left.len(), right.len());
    }
    let (c0, r0) = centroid_radius(xy, &left);
    let (c1, r1) = centroid_radius(xy, &right);
    let gap = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
    let floor = if gap > r0 + r1 { 2 } else { 1 };
    (floor, left.len(), right.len())
}

/// Torgerson 2-D map of DECAF histograms under L1, then
/// [`occupancy_map_floor`].
pub fn occupancy_landfold_floor(histograms: &[Vec<f64>], family: &[usize]) -> usize {
    occupancy_landfold_split(histograms, family).0
}

/// Landfold floor and 2-means side counts of DECAF histograms.
pub fn occupancy_landfold_split(
    histograms: &[Vec<f64>],
    family: &[usize],
) -> (usize, usize, usize) {
    let Some(xy) = occupancy_map_from_histograms(histograms) else {
        return (1, histograms.len(), 0);
    };
    let (floor, left_n, right_n) = occupancy_map_split(&xy, family);
    if floor == 1 {
        return (1, left_n, right_n);
    }
    let Some((left, right)) = two_means(&xy) else {
        return (1, left_n, right_n);
    };
    let Some(left_family) = majority_family(family, &left) else {
        return (1, left_n, right_n);
    };
    let Some(right_family) = majority_family(family, &right) else {
        return (1, left_n, right_n);
    };
    let Some(left_i) = left.iter().copied().find(|&i| family[i] == left_family) else {
        return (1, left_n, right_n);
    };
    let Some(right_i) = right.iter().copied().find(|&i| family[i] == right_family) else {
        return (1, left_n, right_n);
    };
    let between = super::packing::packing_distance(&histograms[left_i], &histograms[right_i]);
    let within = side_spread(histograms, &left).max(side_spread(histograms, &right));
    if between > within {
        (2, left_n, right_n)
    } else {
        (1, left_n, right_n)
    }
}

/// Occupancy free energy. \(F/kT = -\ln(\rho/\rho_{\max})\).
///
/// Discrete packing FES uses leftover-well counts per DECAF family.
/// The landfold-map FES uses a Gaussian KDE on the Torgerson plane.
/// Equal occupancy is \(\Delta F = 0\), not a DECAF L1 split. This
/// does not retire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccupancyFes {
    /// Local minima of \(F/kT\).
    pub minima: usize,
    /// \(\Delta F/kT\) between the two deepest minima.
    pub delta: Option<f64>,
}

/// Discrete packing \(\Delta F/kT = \ln(n_{\max}/n_2)\) from leftover-well
/// counts. None when fewer than two occupied families have wells.
pub fn occupancy_fes_delta(counts: &[u64]) -> Option<f64> {
    let mut occupied: Vec<u64> = counts.iter().copied().filter(|&n| n > 0).collect();
    if occupied.len() < 2 {
        return None;
    }
    occupied.sort_unstable();
    let n_max = occupied[occupied.len() - 1] as f64;
    let n_second = occupied[occupied.len() - 2] as f64;
    if n_max <= 0.0 || n_second <= 0.0 {
        return None;
    }
    Some((n_max / n_second).ln())
}

/// Map free energy from landfold points, optional per-point weights.
///
/// Density is a Gaussian KDE. \(F_i/kT = -\ln(\rho_i/\rho_{\max})\).
/// Minima of \(F\) are maxima of \(\rho\). Bandwidth is
/// \(\max(\mathrm{median\ NN}, 0.25\times\mathrm{diameter})\).
pub fn occupancy_fes(xy: &[[f64; 2]], weights: Option<&[f64]>) -> OccupancyFes {
    let n = xy.len();
    if n < 2 {
        return OccupancyFes {
            minima: 1,
            delta: None,
        };
    }
    let mut nearest = Vec::with_capacity(n);
    let mut diameter = 0.0_f64;
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = map_dist(xy[i], xy[j]);
            best = best.min(d);
            if j > i {
                diameter = diameter.max(d);
            }
        }
        nearest.push(best);
    }
    nearest.sort_by(|a, b| a.total_cmp(b));
    let sigma = nearest[n / 2].max(0.25 * diameter).max(1e-12);
    let mut density = vec![0.0; n];
    for i in 0..n {
        let w_i = weights.and_then(|w| w.get(i).copied()).unwrap_or(1.0);
        if !w_i.is_finite() || w_i <= 0.0 {
            continue;
        }
        for j in 0..n {
            let ratio = map_dist(xy[i], xy[j]) / sigma;
            density[j] += w_i * (-0.5 * ratio * ratio).exp();
        }
    }
    let rho_max = density.iter().copied().fold(0.0_f64, f64::max);
    if rho_max <= 0.0 {
        return OccupancyFes {
            minima: 1,
            delta: None,
        };
    }
    let mut f = vec![0.0; n];
    for i in 0..n {
        if density[i] > 0.0 {
            f[i] = -(density[i] / rho_max).ln();
        } else {
            f[i] = f64::INFINITY;
        }
    }
    let mut minima_f = Vec::new();
    for i in 0..n {
        if !f[i].is_finite() {
            continue;
        }
        let mut low = true;
        for j in 0..n {
            if i == j || map_dist(xy[i], xy[j]) > sigma || !f[j].is_finite() {
                continue;
            }
            if f[j] < f[i] - 1e-15 || ((f[j] - f[i]).abs() <= 1e-15 && j < i) {
                low = false;
                break;
            }
        }
        if low {
            minima_f.push(f[i]);
        }
    }
    minima_f.sort_by(|a, b| a.total_cmp(b));
    let minima = minima_f.len().max(1);
    let delta = if minima_f.len() >= 2 {
        Some(minima_f[1] - minima_f[0])
    } else {
        None
    };
    OccupancyFes { minima, delta }
}

/// Landfold map of DECAF histograms, then [`occupancy_fes`].
pub fn occupancy_fes_from_histograms(histograms: &[Vec<f64>]) -> OccupancyFes {
    match occupancy_map_from_histograms(histograms) {
        Some(xy) => occupancy_fes(&xy, None),
        None => OccupancyFes {
            minima: 1,
            delta: None,
        },
    }
}

fn map_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Secondary family floor from Franzblau primitive-ring profiles.
///
/// One (triangles, squares, pentagons) per occupied DECAF family.
/// Two when two occupied families differ in those counts; one when
/// every occupied family shares a profile, so leftover ico wells of
/// one packing stay one community. This is not the hop-graph Fiedler
/// split and it does not retire.
pub fn occupancy_ring_floor(profiles: &[(usize, usize, usize)]) -> usize {
    occupancy_ring_split(profiles).0
}

/// Ring floor, distinct-profile count, and profile count.
pub fn occupancy_ring_split(profiles: &[(usize, usize, usize)]) -> (usize, usize, usize) {
    let n = profiles.len();
    let mut distinct = std::collections::BTreeSet::new();
    for &profile in profiles {
        distinct.insert(profile);
    }
    let n_distinct = distinct.len();
    let floor = if n_distinct >= 2 { 2 } else { 1 };
    (floor, n_distinct, n)
}

/// Franzblau (tri, sq, pent) at [`crate::structure::RING_CUTOFF_SCALE`]
/// times the structure's median nearest-neighbour distance.
pub fn occupancy_ring_profile(coordinates: &[f64]) -> Option<(usize, usize, usize)> {
    occupancy_ring_census(coordinates).map(|census| census.profile)
}

/// Franzblau census of one structure, or `None` below three atoms.
pub fn occupancy_ring_census(coordinates: &[f64]) -> Option<crate::structure::RingCensus> {
    let n = coordinates.len() / 3;
    if n < 3 || coordinates.len() != 3 * n {
        return None;
    }
    let x = ndarray::ArrayView1::from(coordinates);
    Some(crate::structure::ring_census_nn(x, n))
}

/// L1 of two Franzblau profiles. Leave prefers a larger value.
pub fn ring_novelty(origin: (usize, usize, usize), trial: (usize, usize, usize)) -> usize {
    origin.0.abs_diff(trial.0) + origin.1.abs_diff(trial.1) + origin.2.abs_diff(trial.2)
}

/// Atom weight when leaving `origin`.
///
/// Pentagon-rich occupied packing: break 5-rings (the discrete C5).
/// No pentagons: disturb remaining triangles, the close-packed
/// defects. Uniform incidence is a global scale and does not steer.
pub fn ring_leave_weight(origin: (usize, usize, usize), atom: [u32; 3]) -> f64 {
    if origin.2 > 0 {
        1.0 + f64::from(atom[2])
    } else {
        1.0 + f64::from(atom[0])
    }
}

/// Scale a Cartesian increment so leave-ring atoms move more.
///
/// Champion leftover SOAP does not call this. Occupancy archive
/// holes and packing kicks do. A uniform ring support is a global
/// scale; the hop's RMS cap removes it.
pub fn lens_ring_displacement(coordinates: &[f64], dr: &mut [f64]) {
    if coordinates.len() != dr.len() {
        return;
    }
    let Some(census) = occupancy_ring_census(coordinates) else {
        return;
    };
    let n = census.atom.len();
    if dr.len() != 3 * n {
        return;
    }
    for (i, &incidence) in census.atom.iter().enumerate() {
        let weight = ring_leave_weight(census.profile, incidence);
        dr[3 * i] *= weight;
        dr[3 * i + 1] *= weight;
        dr[3 * i + 2] *= weight;
    }
}

fn side_spread(histograms: &[Vec<f64>], members: &[usize]) -> f64 {
    let mut widest = 0.0;
    for (a, &i) in members.iter().enumerate() {
        for &j in members.iter().skip(a + 1) {
            let d = super::packing::packing_distance(&histograms[i], &histograms[j]);
            if d > widest {
                widest = d;
            }
        }
    }
    widest
}

/// Torgerson (1952) classical MDS of DECAF L1 after a Ceriotti switch.
///
/// Raw L1 stretches a leftover-family chain so 2-means splits the
/// ends. The switch \(F(d)=1-1/(1+(d/\sigma)^2)\) with \(\sigma\) the
/// median pairwise L1 keeps intra-funnel distances short.
pub fn occupancy_map_from_histograms(histograms: &[Vec<f64>]) -> Option<Vec<[f64; 2]>> {
    let n = histograms.len();
    if n < 2 {
        return None;
    }
    let mut dist = vec![vec![0.0; n]; n];
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = super::packing::packing_distance(&histograms[i], &histograms[j]);
            if !d.is_finite() {
                return None;
            }
            dist[i][j] = d;
            dist[j][i] = d;
            if d > 0.0 {
                pairs.push(d);
            }
        }
    }
    let sigma = median(&mut pairs).unwrap_or(super::packing::PACKING_MERGE);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            dist[i][j] = ceriotti_switch(dist[i][j], sigma);
        }
    }
    torgerson_2d(&dist)
}

fn ceriotti_switch(distance: f64, sigma: f64) -> f64 {
    if !(distance.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
        return 0.0;
    }
    1.0 - 1.0 / (1.0 + (distance / sigma).powi(2))
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

fn majority_family(family: &[usize], members: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for &i in members {
        let f = *family.get(i)?;
        let count = members
            .iter()
            .filter(|&&j| family.get(j) == Some(&f))
            .count();
        match best {
            Some((_, held)) if count <= held => {}
            _ => best = Some((f, count)),
        }
    }
    best.map(|(f, _)| f)
}

fn centroid_radius(xy: &[[f64; 2]], members: &[usize]) -> ([f64; 2], f64) {
    let n = members.len().max(1) as f64;
    let mut c = [0.0, 0.0];
    for &i in members {
        c[0] += xy[i][0];
        c[1] += xy[i][1];
    }
    c[0] /= n;
    c[1] /= n;
    let radius = members
        .iter()
        .map(|&i| {
            let dx = xy[i][0] - c[0];
            let dy = xy[i][1] - c[1];
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f64::max);
    (c, radius)
}

fn two_means(xy: &[[f64; 2]]) -> Option<(Vec<usize>, Vec<usize>)> {
    let n = xy.len();
    let (a, b) = farthest_pair(xy)?;
    let mut c0 = xy[a];
    let mut c1 = xy[b];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for _ in 0..16 {
        left.clear();
        right.clear();
        for i in 0..n {
            if dist2(xy[i], c0) <= dist2(xy[i], c1) {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        if left.is_empty() || right.is_empty() {
            return None;
        }
        c0 = mean2(xy, &left);
        c1 = mean2(xy, &right);
    }
    Some((left, right))
}

fn farthest_pair(xy: &[[f64; 2]]) -> Option<(usize, usize)> {
    let n = xy.len();
    if n < 2 {
        return None;
    }
    let mut best = (0, 1);
    let mut best_d = -1.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist2(xy[i], xy[j]);
            if d > best_d {
                best_d = d;
                best = (i, j);
            }
        }
    }
    Some(best)
}

fn mean2(xy: &[[f64; 2]], members: &[usize]) -> [f64; 2] {
    let n = members.len().max(1) as f64;
    let mut c = [0.0, 0.0];
    for &i in members {
        c[0] += xy[i][0];
        c[1] += xy[i][1];
    }
    [c[0] / n, c[1] / n]
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn torgerson_2d(dist: &[Vec<f64>]) -> Option<Vec<[f64; 2]>> {
    let n = dist.len();
    if n < 2 {
        return None;
    }
    let mut d2 = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            d2[i][j] = dist[i][j] * dist[i][j];
        }
    }
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            row_mean[i] += d2[i][j];
            col_mean[j] += d2[i][j];
            total += d2[i][j];
        }
    }
    let nf = n as f64;
    for i in 0..n {
        row_mean[i] /= nf;
        col_mean[i] /= nf;
    }
    total /= nf * nf;
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = -0.5 * (d2[i][j] - row_mean[i] - col_mean[j] + total);
        }
    }
    let mut y = vec![[0.0; 2]; n];
    for k in 0..2 {
        let mut v: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let norm0 = v.iter().map(|z| z * z).sum::<f64>().sqrt().max(1e-15);
        for z in &mut v {
            *z /= norm0;
        }
        if k == 1 {
            let prev: f64 = v.iter().zip(y.iter()).map(|(a, row)| a * row[0]).sum();
            for i in 0..n {
                v[i] -= prev * y[i][0];
            }
            let norm = v.iter().map(|z| z * z).sum::<f64>().sqrt().max(1e-15);
            for z in &mut v {
                *z /= norm;
            }
        }
        let mut lambda = 0.0;
        for _ in 0..80 {
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += b[i][j] * v[j];
                }
            }
            if k == 1 {
                let prev: f64 = w.iter().zip(y.iter()).map(|(a, row)| a * row[0]).sum();
                for i in 0..n {
                    w[i] -= prev * y[i][0];
                }
            }
            lambda = w.iter().zip(v.iter()).map(|(a, c)| a * c).sum();
            let norm = w.iter().map(|z| z * z).sum::<f64>().sqrt().max(1e-15);
            for i in 0..n {
                v[i] = w[i] / norm;
            }
        }
        let scale = lambda.max(0.0).sqrt();
        for i in 0..n {
            y[i][k] = scale * v[i];
        }
    }
    Some(y)
}

/// Occupied-family floor. Spectral split unless `CATALOG_MIN_FAMILIES`
/// is set.
pub fn occupancy_min_families() -> usize {
    std::env::var("CATALOG_MIN_FAMILIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&count| count >= 1)
        .unwrap_or(DEFAULT_MIN_OCCUPIED_FAMILIES)
}

/// Generic occupancy completeness. `n_occupied_families` is the DECAF
/// packing count, not leftover-SOAP basin count.
pub fn occupancy_complete(
    mixing_certified: bool,
    catalog_saturated: bool,
    n_occupied_families: usize,
) -> Option<OccupancyCertificate> {
    occupancy_complete_at(
        mixing_certified,
        catalog_saturated,
        n_occupied_families,
        occupancy_min_families(),
    )
}

/// Completeness at an explicit family floor.
pub fn occupancy_complete_at(
    mixing_certified: bool,
    catalog_saturated: bool,
    n_occupied_families: usize,
    min_occupied_families: usize,
) -> Option<OccupancyCertificate> {
    if mixing_certified {
        Some(OccupancyCertificate::MixingCertified)
    } else if catalog_saturated && n_occupied_families >= min_occupied_families {
        Some(OccupancyCertificate::CatalogSaturated)
    } else {
        None
    }
}

/// Role of one walk against DECAF packing families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingRole {
    /// Lowest energy occupant of this family. Isomer walk.
    FamilyChampion,
    /// Same family, not the champion. Leave or keep-fraction extra.
    FamilyExtra,
    /// No catalog or live walk shares this packing.
    NovelFamily,
}

/// Per-family energy class, not the catalog-wide incumbent.
/// Equal energy to the family best is Champion here; the coordinator
/// keeps one replica via family_champion_replicas so extras Leave.
pub fn packing_role(same_family: bool, energy: f64, best_of_family: Option<f64>) -> PackingRole {
    if !same_family {
        return PackingRole::NovelFamily;
    }
    match best_of_family {
        None => PackingRole::FamilyChampion,
        Some(best) if energy <= best + 1e-8 => PackingRole::FamilyChampion,
        Some(_) => PackingRole::FamilyExtra,
    }
}

/// Champion of the occupied packing has no TIS interface.
pub const CHAMPION_RANK: u32 = u32::MAX;
/// Fixed leftover-SOAP horizon for the TIS ladder. Unit leftover
/// descriptors sit in a ball of diameter 2. Live `max(lambda)` is not
/// a horizon: it collapses every interface onto the well.
pub const INTERFACE_HORIZON: f64 = 2.0;

/// One leftover-SOAP interface seat owned by the catalog RPC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceSeat {
    /// Replica that holds this seat.
    pub replica: u32,
    /// Interface rank. [`CHAMPION_RANK`] is the A-ensemble isomer walk.
    pub rank: u32,
    /// Threshold \(\lambda_i\) this extra must reach.
    pub threshold: f64,
    /// Highest leftover-SOAP \(\lambda\) this replica has posted.
    pub lambda: f64,
}

impl InterfaceSeat {
    /// Occupied-packing champion: no interface, isomer walk only.
    pub fn champion(replica: u32) -> Self {
        Self {
            replica,
            rank: CHAMPION_RANK,
            threshold: 0.0,
            lambda: 0.0,
        }
    }

    /// Extra on interface `rank` with threshold \(\lambda_i\).
    pub fn extra(replica: u32, rank: u32, threshold: f64) -> Self {
        Self {
            replica,
            rank,
            threshold,
            lambda: 0.0,
        }
    }
}

/// Leftover-SOAP order parameter: distance from the occupied-well centroid.
pub fn leftover_lambda(current: &[f64], centroid: &[f64]) -> f64 {
    if current.is_empty() || current.len() != centroid.len() {
        return 0.0;
    }
    current
        .iter()
        .zip(centroid)
        .map(|(value, mean)| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

/// TIS ladder: \(n\) interfaces equally spaced out to `horizon`.
pub fn interface_ladder(n_extras: usize, horizon: f64) -> Vec<f64> {
    if n_extras == 0 || !horizon.is_finite() || horizon <= 0.0 {
        return Vec::new();
    }
    (1..=n_extras)
        .map(|index| horizon * index as f64 / n_extras as f64)
        .collect()
}

/// Whether this sample belongs in interface ensemble \(i\).
pub fn in_interface_ensemble(max_lambda: f64, threshold: f64) -> bool {
    max_lambda.is_finite() && threshold.is_finite() && max_lambda >= threshold
}

/// RETIS swap: each sample already satisfies the other's interface.
pub fn retis_should_swap(lambda_a: f64, thresh_a: f64, lambda_b: f64, thresh_b: f64) -> bool {
    thresh_a != thresh_b
        && in_interface_ensemble(lambda_a, thresh_b)
        && in_interface_ensemble(lambda_b, thresh_a)
}

/// Leave accepts a DECAF family change or an interface crossing.
pub fn leave_shot_accepted(family_changed: bool, trial_lambda: f64, threshold: f64) -> bool {
    family_changed || in_interface_ensemble(trial_lambda, threshold)
}

/// One frame on the occupancy Leave path.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaveFrame {
    /// Cartesian coordinates at this frame.
    pub coordinates: Vec<f64>,
    /// Leftover-SOAP descriptor at this frame.
    pub leftover: Vec<f64>,
    /// Leftover-SOAP \(\lambda\) from the occupied centroid.
    pub lambda: f64,
}

/// Path the extra shoots from. OPS keeps the furthest frame; a failed
/// quench does not throw away the climb toward the interface.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeavePath {
    frames: Vec<LeaveFrame>,
}

impl LeavePath {
    /// Append one checkpoint. The path is a short climbing window.
    pub fn push(&mut self, coordinates: Vec<f64>, leftover: Vec<f64>, lambda: f64) {
        if !lambda.is_finite() {
            return;
        }
        self.frames.push(LeaveFrame {
            coordinates,
            leftover,
            lambda,
        });
        while self.frames.len() > 32 {
            self.frames.remove(0);
        }
    }

    /// Drop the climbing window. Once the Leave is issued the frames
    /// that led to it describe a funnel the chain has left, and a later
    /// shoot taken from them re-enters it.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Highest leftover-SOAP \(\lambda\) on the path.
    pub fn max_lambda(&self) -> f64 {
        self.frames
            .iter()
            .map(|frame| frame.lambda)
            .fold(0.0, f64::max)
    }

    /// Interior frame with the highest \(\lambda\), else the last frame.
    pub fn shoot_index(&self) -> Option<usize> {
        match self.frames.len() {
            0 => None,
            1 | 2 => Some(self.frames.len() - 1),
            n => self.frames[1..n - 1]
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.lambda.total_cmp(&right.1.lambda))
                .map(|(index, _)| index + 1),
        }
    }

    /// Coordinates to shoot from.
    pub fn shoot_coordinates(&self) -> Option<&[f64]> {
        self.shoot_index()
            .map(|index| self.frames[index].coordinates.as_slice())
    }

    /// Leftover-SOAP descriptor at the shoot frame.
    pub fn shoot_leftover(&self) -> Option<&[f64]> {
        self.shoot_index()
            .map(|index| self.frames[index].leftover.as_slice())
    }
}

/// Assign extras increasing leftover-SOAP interfaces. Champion is omitted.
/// Seats are ordered by leftover-SOAP \(\lambda\), not replica id.
pub fn assign_interfaces(extras: &[(u32, f64)], horizon: f64) -> Vec<InterfaceSeat> {
    seat_extras(&[], extras, horizon)
}

/// Seat extras on the TIS ladder, keeping the rank each replica already
/// holds in `previous`. An interface is an ensemble a replica owns until
/// an exchange move hands it on: ranking every request by live
/// \(\lambda\) leaves [`retis_exchange_adjacent`] and
/// [`promote_one_sided`] with an ordering they can never repair, and
/// discards their result on the next request. Unseated extras take the
/// free ranks in \(\lambda\) order, so an empty `previous` is the
/// fresh ladder.
pub fn seat_extras(
    previous: &[InterfaceSeat],
    extras: &[(u32, f64)],
    horizon: f64,
) -> Vec<InterfaceSeat> {
    let ladder = interface_ladder(extras.len(), horizon);
    let mut order = extras.to_vec();
    order.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut holder: Vec<Option<(u32, f64)>> = vec![None; extras.len()];
    let mut seated: Vec<u32> = Vec::new();
    for seat in previous {
        let rank = seat.rank as usize;
        if seat.rank == CHAMPION_RANK || rank >= holder.len() || holder[rank].is_some() {
            continue;
        }
        if seated.contains(&seat.replica) {
            continue;
        }
        let Some(extra) = order.iter().find(|(id, _)| *id == seat.replica) else {
            continue;
        };
        holder[rank] = Some(*extra);
        seated.push(seat.replica);
    }
    let mut free = holder
        .iter()
        .enumerate()
        .filter_map(|(rank, slot)| slot.is_none().then_some(rank))
        .collect::<Vec<_>>()
        .into_iter();
    for extra in &order {
        if seated.contains(&extra.0) {
            continue;
        }
        let Some(rank) = free.next() else {
            break;
        };
        holder[rank] = Some(*extra);
    }
    holder
        .into_iter()
        .enumerate()
        .filter_map(|(rank, slot)| {
            let (replica, lambda) = slot?;
            Some(InterfaceSeat {
                replica,
                rank: rank as u32,
                threshold: ladder.get(rank).copied().unwrap_or(horizon),
                lambda,
            })
        })
        .collect()
}

/// Swap adjacent interface seats when both extras have crossed and the
/// higher \(\lambda\) sits on the lower interface. The ordering guard is
/// what makes the move settle: two samples that each satisfy the other's
/// interface would otherwise trade seats on every request.
pub fn retis_exchange_adjacent(seats: &mut [InterfaceSeat]) -> bool {
    let mut swapped = false;
    let mut index = 0;
    while index + 1 < seats.len() {
        let left = seats[index];
        let right = seats[index + 1];
        if left.rank == CHAMPION_RANK || right.rank == CHAMPION_RANK {
            index += 1;
            continue;
        }
        if retis_should_swap(left.lambda, left.threshold, right.lambda, right.threshold)
            && left.lambda > right.lambda
        {
            seats[index].replica = right.replica;
            seats[index].lambda = right.lambda;
            seats[index + 1].replica = left.replica;
            seats[index + 1].lambda = left.lambda;
            swapped = true;
            index += 2;
        } else {
            index += 1;
        }
    }
    swapped
}

/// Promote an extra that already cleared the next threshold, without
/// waiting for the neighbor to cross this one.
pub fn promote_one_sided(seats: &mut [InterfaceSeat]) -> bool {
    let mut promoted = false;
    let mut index = 0;
    while index + 1 < seats.len() {
        let left = seats[index];
        let right = seats[index + 1];
        if left.rank == CHAMPION_RANK || right.rank == CHAMPION_RANK {
            index += 1;
            continue;
        }
        if in_interface_ensemble(left.lambda, right.threshold)
            && !in_interface_ensemble(right.lambda, left.threshold)
        {
            seats[index].replica = right.replica;
            seats[index].lambda = right.lambda;
            seats[index + 1].replica = left.replica;
            seats[index + 1].lambda = left.lambda;
            promoted = true;
            index += 2;
        } else {
            index += 1;
        }
    }
    promoted
}

/// Replica retire. Mixing certified, packing Good--Turing, leftover
/// dwell, FunnelModel EI exhausted on seen packings, and the rematched
/// family floor. CatalogSaturated names census completeness and does
/// not retire: extras keep Leaving.
/// `catalog_saturated` is packing-family saturation, not leftover-SOAP.
/// `leftover_dwell` is consecutive leftover-sat occupancy_gt records,
/// not a one-shot leftover nick. Required whenever the packing book
/// holds more than one family. A single book family is Boender--Rinnooy
/// Kan on one cell: leftover-SOAP hatches are intra-well and do not
/// block. Live rematch of last candidates after extras Leave is not
/// that count. A one-community Fiedler floor with many DECAF packings
/// is not that case.
/// `ei_exhausted` is Jones remaining improvement on observed
/// FunnelModel morphologies, not a far-field GP probe.
/// `n_occupied_families` is the packing-book occupied-family count
/// (visits > 0), not live rematch of last candidates and not a
/// leftover-SOAP basin count. One book family is Boender--Rinnooy Kan
/// on one cell. Many book families with extras already Left is not.
pub fn occupancy_retire(
    certificate: OccupancyCertificate,
    catalog_saturated: bool,
    leftover_dwell: bool,
    ei_exhausted: bool,
    n_occupied_families: usize,
) -> bool {
    occupancy_retire_at(
        certificate,
        catalog_saturated,
        leftover_dwell,
        ei_exhausted,
        n_occupied_families,
        occupancy_min_families(),
    )
}

/// Retire at an explicit family floor. Good--Turing alone is floor 1.
pub fn occupancy_retire_at(
    certificate: OccupancyCertificate,
    catalog_saturated: bool,
    leftover_dwell: bool,
    ei_exhausted: bool,
    n_occupied_families: usize,
    min_occupied_families: usize,
) -> bool {
    n_occupied_families >= min_occupied_families
        && catalog_saturated
        && (leftover_dwell || n_occupied_families <= 1)
        && ei_exhausted
        && matches!(certificate, OccupancyCertificate::MixingCertified)
}

#[cfg(test)]
mod tests {
    use super::{
        CHAMPION_RANK, InterfaceSeat, LeavePath, OCCUPANCY_SEAM_CONDUCTANCE, OccupancyCertificate,
        OccupancyLeaveAdopt, OccupancyLeaveTarget, PackingRole, assign_interfaces,
        in_interface_ensemble, interface_ladder, is_occupancy_leave_action, leave_shot_accepted,
        leftover_lambda, leftover_sat_dwell, occupancy_complete, occupancy_complete_at,
        occupancy_ei_exhausted, occupancy_family_floor, occupancy_fes, occupancy_fes_delta,
        occupancy_fes_from_histograms, occupancy_landfold_floor, occupancy_leave_adopt,
        occupancy_leave_target, occupancy_map_floor, occupancy_retire, occupancy_retire_at,
        occupancy_ring_floor, packing_role, promote_one_sided, published_energy_score,
        retis_exchange_adjacent, retis_should_swap, ring_leave_weight, ring_novelty, seat_extras,
    };

    #[test]
    fn feynman_kac_extras_use_the_same_leave_as_occupancy() {
        assert!(is_occupancy_leave_action("population_reseed"));
        assert!(is_occupancy_leave_action("hyperband_reseed"));
        assert!(is_occupancy_leave_action("catalog_leave"));
        assert!(!is_occupancy_leave_action("catalog_incumbent"));
        assert!(!is_occupancy_leave_action("bridge"));
    }

    #[test]
    fn occupancy_leave_is_another_family_or_an_archive_hole() {
        assert_eq!(
            occupancy_leave_target(true, false),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, false),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_ne!(
            occupancy_leave_target(false, false),
            OccupancyLeaveTarget::OtherFamily
        );
    }

    #[test]
    fn packing_sat_leave_is_archive_hole_not_other_family() {
        assert_eq!(
            occupancy_leave_target(true, true),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_ne!(
            occupancy_leave_target(true, true),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, true),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_eq!(
            occupancy_leave_target(true, false),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, false),
            OccupancyLeaveTarget::ArchiveHole
        );
    }

    #[test]
    fn catalog_leave_refuses_a_same_family_hole() {
        assert_eq!(
            occupancy_leave_adopt("catalog_leave", false),
            Some(OccupancyLeaveAdopt::Refuse)
        );
        assert_eq!(
            occupancy_leave_adopt("catalog_leave", true),
            Some(OccupancyLeaveAdopt::Quench)
        );
        assert_eq!(
            occupancy_leave_adopt("hyperband_reseed", false),
            Some(OccupancyLeaveAdopt::Quench)
        );
        assert_eq!(
            occupancy_leave_adopt("population_reseed", false),
            Some(OccupancyLeaveAdopt::Quench)
        );
        assert_eq!(occupancy_leave_adopt("catalog_incumbent", false), None);
        assert_eq!(occupancy_leave_adopt("histo", true), None);
    }

    #[test]
    fn a_published_energy_is_a_score_not_a_certificate() {
        assert!(published_energy_score(-173.928427, Some(-173.928427)));
        assert!(published_energy_score(-397.492331, Some(-397.492331)));
        assert!(!published_energy_score(-396.282249, Some(-397.492331)));
        assert!(!published_energy_score(-173.928427, None));
        assert_eq!(occupancy_complete(false, false, 0), None);
    }

    #[test]
    fn inverted_gr_mixing_certifies_the_search() {
        assert_eq!(
            occupancy_complete(true, false, 1),
            Some(OccupancyCertificate::MixingCertified)
        );
    }

    #[test]
    fn packing_saturation_below_the_family_floor_is_not_done() {
        assert_eq!(occupancy_complete_at(false, true, 1, 2), None);
        assert_eq!(occupancy_complete_at(false, true, 0, 2), None);
    }

    #[test]
    fn packing_good_turing_with_no_new_families_is_saturated() {
        assert_eq!(
            occupancy_complete_at(false, true, 1, 1),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert_eq!(occupancy_complete_at(false, false, 5, 1), None);
    }

    #[test]
    fn packing_gt_with_two_families_meets_the_paper_floor() {
        assert_eq!(
            occupancy_complete(false, true, 2),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            1,
            2
        ));
    }

    #[test]
    fn catalog_saturation_does_not_retire_without_mixing() {
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2
        ));
    }

    #[test]
    fn packing_role_is_per_family_not_catalog_wide() {
        assert_eq!(
            packing_role(true, -173.252378, Some(-173.252378)),
            PackingRole::FamilyChampion
        );
        assert_eq!(
            packing_role(true, -173.134317, Some(-173.252378)),
            PackingRole::FamilyExtra
        );
        assert_eq!(
            packing_role(false, -173.928427, Some(-173.252378)),
            PackingRole::NovelFamily
        );
        assert_eq!(
            packing_role(true, -173.928427, Some(-173.252378)),
            PackingRole::FamilyChampion
        );
    }

    #[test]
    fn good_turing_with_a_competing_family_is_a_certificate() {
        assert_eq!(
            occupancy_complete(false, true, 2),
            Some(OccupancyCertificate::CatalogSaturated)
        );
    }

    #[test]
    fn mixing_outranks_catalog_saturation() {
        assert_eq!(
            occupancy_complete(true, true, 2),
            Some(OccupancyCertificate::MixingCertified)
        );
    }

    #[test]
    fn mixing_alone_does_not_retire_extras_before_good_turing() {
        assert!(!occupancy_retire(
            OccupancyCertificate::MixingCertified,
            false,
            true,
            true,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            true,
            1,
            2
        ));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            1,
            2
        ));
        assert_eq!(occupancy_complete(false, false, 8), None);
    }

    #[test]
    fn leftover_unsaturated_does_not_retire_with_mixing_and_packing_sat() {
        assert!(!leftover_sat_dwell(&[false]));
        assert!(!leftover_sat_dwell(&[true]));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            false,
            true,
            2,
            2
        ));
    }

    #[test]
    fn leftover_nick_is_not_a_retire_dwell() {
        assert!(!leftover_sat_dwell(&[true, false]));
        assert!(!leftover_sat_dwell(&[false, true]));
        assert!(!leftover_sat_dwell(&[true; 4]));
        assert!(leftover_sat_dwell(&[true; 5]));
        let mut hatch_then_dwell = vec![false];
        hatch_then_dwell.extend(std::iter::repeat_n(true, 5));
        assert!(leftover_sat_dwell(&hatch_then_dwell));
        let mut dwell_then_hatch = vec![true; 5];
        dwell_then_hatch.push(false);
        assert!(!leftover_sat_dwell(&dwell_then_hatch));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            leftover_sat_dwell(&[true]),
            true,
            2,
            2
        ));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            1,
            1
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            34,
            1
        ));
    }

    #[test]
    fn leftover_dwell_plus_mixing_plus_packing_sat_plus_floor_retires() {
        assert!(leftover_sat_dwell(&[true; 5]));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            leftover_sat_dwell(&[true; 5]),
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            false,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            true,
            1,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            false,
            2,
            2
        ));
    }

    #[test]
    fn funnel_ei_below_noise_with_three_observations_is_exhausted() {
        assert!(!occupancy_ei_exhausted(1.0, 3, 1e-2));
        assert!(!occupancy_ei_exhausted(0.0, 2, 1e-2));
        assert!(occupancy_ei_exhausted(0.0, 3, 1e-2));
        assert!(occupancy_ei_exhausted(1e-2, 3, 1e-2));
        assert!(!occupancy_ei_exhausted(f64::INFINITY, 3, 1e-2));
    }

    #[test]
    fn map_floor_keeps_one_packing_together() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [8.0, 8.0]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 0, 0]), 1);
    }

    #[test]
    fn map_floor_splits_two_separated_packings() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 1, 1]), 2);
    }

    #[test]
    fn landfold_floor_on_identical_histograms_is_one() {
        let ico = vec![1.0, 0.0];
        assert_eq!(
            occupancy_landfold_floor(&[ico.clone(), ico.clone()], &[0, 0]),
            1
        );
    }

    #[test]
    fn landfold_floor_on_separated_histograms_is_two() {
        let ico = vec![1.0, 0.0];
        let oh = vec![0.0, 1.0];
        assert_eq!(occupancy_landfold_floor(&[ico, oh], &[0, 1]), 2);
    }

    #[test]
    fn ring_floor_on_one_profile_is_one() {
        assert_eq!(occupancy_ring_floor(&[]), 1);
        assert_eq!(occupancy_ring_floor(&[(165, 17, 6)]), 1);
        assert_eq!(occupancy_ring_floor(&[(165, 17, 6), (165, 17, 6)]), 1);
    }

    #[test]
    fn ring_floor_on_distinct_profiles_is_two() {
        assert_eq!(occupancy_ring_floor(&[(165, 17, 6), (152, 22, 0)]), 2);
        assert_eq!(occupancy_ring_floor(&[(400, 42, 21), (360, 55, 2)]), 2);
    }

    #[test]
    fn packing_fes_delta_is_log_occupancy_ratio() {
        assert_eq!(occupancy_fes_delta(&[]), None);
        assert_eq!(occupancy_fes_delta(&[20]), None);
        assert_eq!(occupancy_fes_delta(&[0, 0]), None);
        let delta = occupancy_fes_delta(&[20, 5]).unwrap();
        assert!((delta - 4.0_f64.ln()).abs() < 1e-12);
        let even = occupancy_fes_delta(&[10, 10]).unwrap();
        assert!(even.abs() < 1e-12);
    }

    #[test]
    fn fes_map_is_free_energy_not_a_decaf_split() {
        let chain: Vec<[f64; 2]> = (0..8).map(|i| [i as f64, 0.0]).collect();
        let chain_fes = occupancy_fes(&chain, None);
        assert_eq!(chain_fes.minima, 1);
        assert!(chain_fes.delta.is_none());
        let clumps = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        let clump_fes = occupancy_fes(&clumps, None);
        assert_eq!(clump_fes.minima, 2);
        assert!(clump_fes.delta.is_some());
        let one = occupancy_fes_from_histograms(&[vec![1.0, 0.0]]);
        assert_eq!(one.minima, 1);
        assert!(one.delta.is_none());
    }

    #[test]
    fn ring_leave_weight_breaks_pentagons_on_ico_and_triangles_on_oh() {
        let ico = (165, 17, 6);
        let oh = (152, 22, 0);
        assert!(ring_leave_weight(ico, [0, 0, 3]) > ring_leave_weight(ico, [10, 0, 0]));
        assert!(ring_leave_weight(oh, [4, 0, 0]) > ring_leave_weight(oh, [0, 2, 0]));
        assert_eq!(ring_novelty(ico, ico), 0);
        assert!(ring_novelty(ico, oh) > 0);
    }

    #[test]
    fn spectral_seam_is_two_communities_otherwise_one() {
        assert_eq!(occupancy_family_floor(None, 0, 0, false), 1);
        assert_eq!(occupancy_family_floor(Some(0.2), 4, 5, true), 1);
        assert_eq!(occupancy_family_floor(Some(0.05), 4, 0, true), 1);
        assert_eq!(occupancy_family_floor(Some(0.05), 4, 5, false), 1);
        assert_eq!(occupancy_family_floor(Some(0.05), 4, 5, true), 2);
        assert_eq!(
            occupancy_family_floor(Some(OCCUPANCY_SEAM_CONDUCTANCE), 4, 5, true),
            1
        );
    }

    #[test]
    fn a_user_family_floor_of_one_is_good_turing_alone() {
        assert_eq!(
            occupancy_complete_at(false, true, 1, 1),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            1,
            1
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2,
            3
        ));
    }

    #[test]
    fn leftover_lambda_is_the_distance_from_the_occupied_centroid() {
        let centroid = [0.0, 0.0, 0.0];
        let on_well = [0.0, 0.0, 0.0];
        let off_well = [3.0, 4.0, 0.0];
        assert_eq!(leftover_lambda(&on_well, &centroid), 0.0);
        assert_eq!(leftover_lambda(&off_well, &centroid), 5.0);
        assert_eq!(leftover_lambda(&[1.0], &centroid), 0.0);
    }

    #[test]
    fn interface_ladder_stages_extras_away_from_the_occupied_well() {
        assert!(interface_ladder(0, 1.0).is_empty());
        assert_eq!(interface_ladder(2, 2.0), vec![1.0, 2.0]);
        assert_eq!(interface_ladder(4, 1.0), vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn a_shot_is_in_the_interface_ensemble_once_it_reaches_the_threshold() {
        assert!(!in_interface_ensemble(0.4, 0.5));
        assert!(in_interface_ensemble(0.5, 0.5));
        assert!(in_interface_ensemble(0.9, 0.5));
    }

    #[test]
    fn retis_swaps_adjacent_ranks_only_when_each_sample_satisfies_the_other() {
        assert!(retis_should_swap(1.0, 0.25, 0.6, 0.5));
        assert!(!retis_should_swap(0.4, 0.25, 0.6, 0.5));
        assert!(!retis_should_swap(1.0, 0.5, 1.0, 0.5));
        assert!(!retis_should_swap(1.0, 0.25, 0.2, 0.5));
    }

    #[test]
    fn leave_accepts_a_family_change_or_an_interface_crossing() {
        assert!(leave_shot_accepted(true, 0.1, 0.5));
        assert!(leave_shot_accepted(false, 0.5, 0.5));
        assert!(!leave_shot_accepted(false, 0.4, 0.5));
    }

    #[test]
    fn a_leave_path_shoots_from_the_highest_interior_lambda() {
        let mut path = LeavePath::default();
        path.push(vec![0.0], vec![0.0], 0.1);
        path.push(vec![1.0], vec![1.0], 0.8);
        path.push(vec![2.0], vec![2.0], 0.3);
        assert_eq!(path.max_lambda(), 0.8);
        assert_eq!(path.shoot_index(), Some(1));
        assert_eq!(path.shoot_coordinates(), Some([1.0].as_slice()));
    }

    #[test]
    fn a_cleared_leave_path_has_no_shoot_frame() {
        let mut path = LeavePath::default();
        path.push(vec![0.0], vec![0.0], 0.1);
        path.push(vec![1.0], vec![1.0], 0.8);
        path.clear();
        assert_eq!(path.shoot_index(), None);
        assert_eq!(path.shoot_coordinates(), None);
        assert_eq!(path.max_lambda(), 0.0);
    }

    #[test]
    fn interface_ranks_follow_lambda_not_replica_id() {
        let extras = [(3, 0.1), (1, 0.9), (2, 0.4)];
        let assigned = assign_interfaces(&extras, 1.0);
        assert_eq!(assigned[0].replica, 3);
        assert_eq!(assigned[0].rank, 0);
        assert_eq!(assigned[1].replica, 2);
        assert_eq!(assigned[2].replica, 1);
        assert!((assigned[2].threshold - 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_sided_promotion_does_not_wait_for_the_neighbor() {
        let mut seats = vec![
            InterfaceSeat {
                replica: 1,
                rank: 0,
                threshold: 0.5,
                lambda: 1.0,
            },
            InterfaceSeat {
                replica: 2,
                rank: 1,
                threshold: 1.0,
                lambda: 0.2,
            },
        ];
        let mut unexchanged = seats.clone();
        assert!(!retis_exchange_adjacent(&mut unexchanged));
        assert!(promote_one_sided(&mut seats));
        assert_eq!(seats[1].replica, 1);
        assert_eq!(seats[0].replica, 2);
        assert!(in_interface_ensemble(seats[1].lambda, seats[1].threshold));
        assert!(!promote_one_sided(&mut seats));
    }

    #[test]
    fn extras_receive_increasing_interface_ranks_and_the_champion_does_not() {
        let extras = [(1, 0.1), (2, 0.4), (3, 0.9)];
        let assigned = assign_interfaces(&extras, 1.0);
        assert_eq!(assigned.len(), 3);
        assert_eq!(assigned[0].replica, 1);
        assert_eq!(assigned[0].rank, 0);
        assert!((assigned[0].threshold - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(assigned[2].replica, 3);
        assert!((assigned[2].threshold - 1.0).abs() < 1e-12);
        assert_eq!(InterfaceSeat::champion(0).rank, CHAMPION_RANK);
    }

    #[test]
    fn retis_exchanges_adjacent_seats_when_both_have_crossed() {
        let mut seats = vec![
            InterfaceSeat {
                replica: 1,
                rank: 0,
                threshold: 0.5,
                lambda: 1.0,
            },
            InterfaceSeat {
                replica: 2,
                rank: 1,
                threshold: 1.0,
                lambda: 0.8,
            },
        ];
        assert!(retis_exchange_adjacent(&mut seats));
        assert_eq!(seats[0].replica, 2);
        assert_eq!(seats[1].replica, 1);
        assert!(!retis_exchange_adjacent(&mut seats));
    }

    #[test]
    fn a_seated_extra_keeps_its_interface_when_lambda_moves() {
        let first = assign_interfaces(&[(1, 0.1), (2, 0.4), (3, 0.9)], 1.0);
        assert_eq!(first[0].replica, 1);
        assert_eq!(first[2].replica, 3);
        let climbed = seat_extras(&first, &[(1, 0.95), (2, 0.4), (3, 0.9)], 1.0);
        assert_eq!(climbed[0].replica, 1);
        assert_eq!(climbed[0].rank, 0);
        assert!((climbed[0].lambda - 0.95).abs() < 1e-12);
        assert_eq!(climbed[2].replica, 3);
    }

    #[test]
    fn a_new_extra_takes_the_rank_a_departed_one_freed() {
        let first = assign_interfaces(&[(1, 0.1), (2, 0.4)], 1.0);
        assert_eq!(first[1].replica, 2);
        let swapped = seat_extras(&first, &[(1, 0.1), (7, 0.4)], 1.0);
        assert_eq!(swapped[0].replica, 1);
        assert_eq!(swapped[1].replica, 7);
        assert_eq!(swapped[1].rank, 1);
    }

    #[test]
    fn a_shrunk_ladder_reseats_every_extra() {
        let first = assign_interfaces(&[(1, 0.1), (2, 0.4), (3, 0.9)], 1.0);
        let shrunk = seat_extras(&first, &[(3, 0.9)], 1.0);
        assert_eq!(shrunk.len(), 1);
        assert_eq!(shrunk[0].replica, 3);
        assert_eq!(shrunk[0].rank, 0);
        assert!((shrunk[0].threshold - 1.0).abs() < 1e-12);
    }
}
