//! Quality-diversity archive primitives.
//!
//! The occupancy catalog keeps one best structure per packing family
//! and draws new starts from it, which is MAP-Elites with a learned
//! descriptor in place of a designed one. Three pieces of that
//! literature are missing here and are supplied below, each as
//! arithmetic over descriptors so nothing in this module needs a
//! search, a socket, or a potential.
//!
//! ## Why a tessellation
//!
//! Families are discovered by leader clustering, so their number is
//! whatever the data produced: a campaign reported twenty-two families
//! across twenty-four replicas, which is nearly one cell per replica
//! and not an archive at all. Worse, a saturation statistic over a
//! support that grows as it samples is answering a moving question.
//! Tessellating the descriptor space instead makes the cell count a
//! parameter and the cells equal by construction, which is what
//! centroidal Voronoi tessellations are for in this setting.
//!
//! The tessellation needs a descriptor of fixed width. A DECAF class
//! histogram grows a column whenever the codebook grows one, so it is
//! the wrong input; the fixed-width cloud mean is the right one. That
//! constraint is real and is the reason this is a separate space from
//! the family key rather than a replacement for it.

use rand::Rng;

/// An archive of descriptor cells with an annealed radius.
///
/// Cells are not a grid and not discovered at a fixed radius. A
/// structure joins the nearest cell within `Dcut` and opens a new one
/// beyond it, and `Dcut` anneals from half the mean pairwise distance
/// down to a floor over the run, which is conformational space
/// annealing's rule and [`crate::diversity::DiversityAnnealer`] is
/// already the schedule for it. Early on the archive is coarse and the
/// search is asked only to be different; late on it is fine and the
/// search is asked to be better. That is the simulated-annealing half
/// of CSA applied to diversity rather than to acceptance.
///
/// Selection is the exploration half, and the reference is not the
/// physics literature. Hard-exploration Atari was solved by keeping an
/// archive of visited cells, returning to a promising one, and
/// exploring from there: first return, then explore. A Leave is that
/// return, and the cell it returns to is chosen the same way, by
/// weighting a cell's demonstrated success against how little it has
/// been tried.
pub struct Archive {
    /// Representative descriptor of each cell.
    centres: Vec<Vec<f64>>,
    /// Times a structure landed in each cell.
    visits: Vec<u64>,
    /// Successes plus one: starts from this cell the catalog kept.
    alpha: Vec<f64>,
    /// Failures plus one.
    beta: Vec<f64>,
    /// The annealed radius that decides what counts as a new cell.
    radius: crate::diversity::DiversityAnnealer,
}

impl Archive {
    /// An archive whose radius starts at `initial` and anneals to
    /// `floor_fraction` of it.
    pub fn new(initial: f64, floor_fraction: f64) -> Self {
        Self {
            centres: Vec::new(),
            visits: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
            radius: crate::diversity::DiversityAnnealer::from_initial(initial)
                .with_final_fraction(floor_fraction),
        }
    }

    /// Cells opened so far.
    pub fn cells(&self) -> usize {
        self.centres.len()
    }

    /// Current radius.
    pub fn radius(&self) -> f64 {
        self.radius.current()
    }

    /// Advance the radius to where `progress` through the run puts it.
    pub fn anneal(&mut self, progress: f64) -> f64 {
        self.radius.threshold(progress)
    }

    /// Cell this descriptor belongs to, opening one if it is further
    /// than the radius from every cell already open.
    pub fn observe(&mut self, descriptor: &[f64]) -> Option<usize> {
        if descriptor.is_empty() {
            return None;
        }
        let radius = self.radius.current();
        let nearest = self
            .centres
            .iter()
            .enumerate()
            .filter(|(_, centre)| centre.len() == descriptor.len())
            .map(|(index, centre)| (index, distance(centre, descriptor)))
            .min_by(|left, right| left.1.total_cmp(&right.1));
        match nearest {
            Some((index, gap)) if gap <= radius => {
                self.visits[index] = self.visits[index].saturating_add(1);
                Some(index)
            }
            _ => {
                self.centres.push(descriptor.to_vec());
                self.visits.push(1);
                self.alpha.push(1.0);
                self.beta.push(1.0);
                Some(self.centres.len() - 1)
            }
        }
    }

    /// Cell this descriptor belongs to without opening one.
    pub fn assign(&self, descriptor: &[f64]) -> Option<usize> {
        let radius = self.radius.current();
        self.centres
            .iter()
            .enumerate()
            .filter(|(_, centre)| centre.len() == descriptor.len())
            .map(|(index, centre)| (index, distance(centre, descriptor)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .filter(|(_, gap)| *gap <= radius)
            .map(|(index, _)| index)
    }

    /// Posterior mean success rate of a cell.
    pub fn score(&self, cell: usize) -> f64 {
        let alpha = self.alpha.get(cell).copied().unwrap_or(1.0);
        let beta = self.beta.get(cell).copied().unwrap_or(1.0);
        alpha / (alpha + beta)
    }

    /// Times a structure landed in a cell.
    pub fn visits(&self, cell: usize) -> u64 {
        self.visits.get(cell).copied().unwrap_or(0)
    }

    /// A start drawn from this cell produced something the catalog kept.
    pub fn reward(&mut self, cell: usize) {
        if let Some(alpha) = self.alpha.get_mut(cell) {
            *alpha += 1.0;
        }
    }

    /// A start drawn from this cell produced nothing.
    pub fn penalise(&mut self, cell: usize) {
        if let Some(beta) = self.beta.get_mut(cell) {
            *beta += 1.0;
        }
    }

    /// Return to a cell: a Thompson draw on what the cell has produced,
    /// discounted by how heavily it has already been visited.
    ///
    /// The posterior is the exploit term and the count is the explore
    /// term, which is the shape of the cell-selection rule that made
    /// return-then-explore work: a cell nothing has visited is worth
    /// returning to even with no evidence, and a cell visited a hundred
    /// times has to keep earning it.
    pub fn select<R: Rng + ?Sized>(&self, allowed: &[usize], rng: &mut R) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for cell in allowed {
            let alpha = self.alpha.get(*cell).copied().unwrap_or(1.0);
            let beta = self.beta.get(*cell).copied().unwrap_or(1.0);
            let promise = sample_beta(alpha, beta, rng);
            let seen = self.visits.get(*cell).copied().unwrap_or(0) as f64;
            let bonus = 1.0 / (1.0 + seen).sqrt();
            let weight = promise * bonus;
            if best.is_none_or(|(_, held)| weight > held) {
                best = Some((*cell, weight));
            }
        }
        best.map(|(cell, _)| cell)
    }

    /// Fraction of open cells that any of `descriptors` occupies.
    pub fn coverage<'a, I>(&self, descriptors: I) -> f64
    where
        I: IntoIterator<Item = &'a [f64]>,
    {
        if self.centres.is_empty() {
            return 0.0;
        }
        let mut seen = vec![false; self.centres.len()];
        for descriptor in descriptors {
            if let Some(cell) = self.assign(descriptor) {
                seen[cell] = true;
            }
        }
        seen.iter().filter(|reached| **reached).count() as f64 / self.centres.len() as f64
    }
}

fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Per-cell Beta-Bernoulli bandit over the archive.
///
/// Choosing which cell to draw a start from is a bandit problem and
/// deserves to be treated as one. The reward is binary, the catalog
/// kept what came back or it did not, so the conjugate model is Beta
/// over a Bernoulli rate and the selection rule is Thompson sampling:
/// draw a rate from each cell's posterior and take the highest.
///
/// That is better than the reward-and-decay heuristic it replaces on
/// three counts. It has no constants to pick, where the heuristic had
/// a decay and a floor chosen by hand. Its exploration is automatic:
/// a cell tried twice has a wide posterior and still wins draws, while
/// a cell tried two hundred times does not, so effort moves off a cell
/// only once there is evidence to move it. And it cannot write a cell
/// off, because a Beta posterior never reaches zero, which matters
/// because a descriptor can be wrong where a cell is right.
///
/// The allocator over move kernels in [`crate::allocate`] is the same
/// idea over a Gaussian reward; this is the Bernoulli case.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curiosity {
    /// Successes plus one: starts from this cell the catalog kept.
    alpha: Vec<f64>,
    /// Failures plus one: starts it did not keep.
    beta: Vec<f64>,
}

impl Curiosity {
    /// A bandit over `cells`, each with a uniform prior.
    pub fn new(cells: usize) -> Self {
        Self {
            alpha: vec![1.0; cells],
            beta: vec![1.0; cells],
        }
    }

    /// Grow the table so `cells` are armed, new ones uniform.
    ///
    /// Cells are discovered as the search runs, so the table cannot be
    /// sized once. Growing never disturbs a posterior already earned.
    pub fn ensure(&mut self, cells: usize) {
        if cells > self.alpha.len() {
            self.alpha.resize(cells, 1.0);
            self.beta.resize(cells, 1.0);
        }
    }

    /// Posterior mean success rate of a cell, or the uniform prior for
    /// one the table has not armed.
    pub fn score(&self, cell: usize) -> f64 {
        let alpha = self.alpha.get(cell).copied().unwrap_or(1.0);
        let beta = self.beta.get(cell).copied().unwrap_or(1.0);
        alpha / (alpha + beta)
    }

    /// Times this cell has been drawn from.
    pub fn draws(&self, cell: usize) -> f64 {
        let alpha = self.alpha.get(cell).copied().unwrap_or(1.0);
        let beta = self.beta.get(cell).copied().unwrap_or(1.0);
        alpha + beta - 2.0
    }

    /// A start drawn from this cell produced something the catalog kept.
    pub fn reward(&mut self, cell: usize) {
        if let Some(alpha) = self.alpha.get_mut(cell) {
            *alpha += 1.0;
        }
    }

    /// A start drawn from this cell produced nothing.
    pub fn penalise(&mut self, cell: usize) {
        if let Some(beta) = self.beta.get_mut(cell) {
            *beta += 1.0;
        }
    }

    /// Thompson draw: sample a rate from each allowed cell's posterior
    /// and take the highest.
    pub fn select<R: Rng + ?Sized>(&self, allowed: &[usize], rng: &mut R) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for cell in allowed {
            let alpha = self.alpha.get(*cell).copied().unwrap_or(1.0);
            let beta = self.beta.get(*cell).copied().unwrap_or(1.0);
            let sampled = sample_beta(alpha, beta, rng);
            if best.is_none_or(|(_, held)| sampled > held) {
                best = Some((*cell, sampled));
            }
        }
        best.map(|(cell, _)| cell)
    }
}

/// One Beta draw, as the ratio of two Gammas.
fn sample_beta<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// One Gamma draw by Marsaglia and Tsang, with the shape boost that
/// carries shapes below one.
fn sample_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    if !shape.is_finite() || shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        let boosted = sample_gamma(shape + 1.0, rng);
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        return boosted * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box-Muller for the standard normal the method needs.
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = 1.0 + c * normal;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        if u.ln() < 0.5 * normal * normal + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Novelty of a descriptor: mean distance to its `k` nearest
/// neighbours among those already seen.
///
/// On a deceptive landscape the objective's gradient points into the
/// trap, and this is the quantity a search follows instead. An empty
/// neighbourhood is maximally novel and answers infinity, which is the
/// honest value: nothing has been seen to compare against.
pub fn novelty(descriptor: &[f64], seen: &[Vec<f64>], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let mut distances: Vec<f64> = seen
        .iter()
        .filter(|other| other.len() == descriptor.len())
        .map(|other| {
            other
                .iter()
                .zip(descriptor)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    if distances.is_empty() {
        return f64::INFINITY;
    }
    distances.sort_by(f64::total_cmp);
    let take = k.min(distances.len());
    distances.iter().take(take).sum::<f64>() / take as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn the_radius_anneals_from_coarse_to_fine() {
        let mut archive = Archive::new(1.0, 0.4);
        assert!((archive.radius() - 1.0).abs() < 1e-12);
        archive.anneal(1.0);
        // The floor is a fraction of the start, so late in a run the
        // archive separates what it merged early: different first,
        // better second.
        assert!(
            archive.radius() <= 0.4 + 1e-9,
            "radius {}",
            archive.radius()
        );
        assert!(archive.radius() > 0.0);
    }

    #[test]
    fn a_far_structure_opens_a_cell_and_a_near_one_joins() {
        let mut archive = Archive::new(1.0, 0.4);
        assert_eq!(archive.observe(&[0.0, 0.0]), Some(0));
        // Inside the radius: same cell, and the visit is counted.
        assert_eq!(archive.observe(&[0.2, 0.0]), Some(0));
        assert_eq!(archive.visits(0), 2);
        // Beyond it: a cell of its own.
        assert_eq!(archive.observe(&[5.0, 0.0]), Some(1));
        assert_eq!(archive.cells(), 2);
        assert_eq!(archive.assign(&[0.1, 0.0]), Some(0));
        // Nowhere near anything open, and assign does not open cells.
        assert_eq!(archive.assign(&[50.0, 0.0]), None);
        assert_eq!(archive.cells(), 2);
    }

    #[test]
    fn a_shrinking_radius_splits_what_it_had_merged() {
        let mut archive = Archive::new(1.0, 0.1);
        archive.observe(&[0.0, 0.0]);
        // Half a unit away joins while the radius is one.
        assert_eq!(archive.observe(&[0.5, 0.0]), Some(0));
        archive.anneal(1.0);
        // Once the radius is a tenth it does not, which is the point of
        // annealing it rather than fixing it.
        assert_eq!(archive.observe(&[0.5, 0.0]), Some(1));
    }

    #[test]
    fn an_unvisited_cell_is_worth_returning_to() {
        let mut archive = Archive::new(1.0, 0.4);
        archive.observe(&[0.0, 0.0]);
        archive.observe(&[5.0, 0.0]);
        // Cell zero visited heavily and paying; cell one barely seen.
        for _ in 0..60 {
            archive.observe(&[0.05, 0.0]);
            archive.reward(0);
        }
        let mut rng = StdRng::seed_from_u64(31);
        let returned = (0..400)
            .filter(|_| archive.select(&[0, 1], &mut rng) == Some(1))
            .count();
        // The count discount is what makes a thin cell worth a return
        // even against a cell with a far better record: sixty visits
        // against one is a bonus ratio near eight.
        assert!(
            returned > 100,
            "barely-visited cell returned to only {returned} of 400"
        );
    }

    #[test]
    fn a_paying_cell_still_wins_against_an_equally_thin_one() {
        let mut archive = Archive::new(1.0, 0.4);
        archive.observe(&[0.0, 0.0]);
        archive.observe(&[5.0, 0.0]);
        for _ in 0..30 {
            archive.reward(0);
            archive.penalise(1);
        }
        let mut rng = StdRng::seed_from_u64(37);
        let picked = (0..300)
            .filter(|_| archive.select(&[0, 1], &mut rng) == Some(0))
            .count();
        assert!(picked > 250, "paying cell chosen only {picked} of 300");
    }

    #[test]
    fn coverage_counts_the_cells_the_ensemble_stands_on() {
        let mut archive = Archive::new(1.0, 0.4);
        archive.observe(&[0.0, 0.0]);
        archive.observe(&[5.0, 0.0]);
        archive.observe(&[10.0, 0.0]);
        let here = [0.0, 0.0];
        let covered = archive.coverage(vec![&here[..]]);
        assert!((covered - 1.0 / 3.0).abs() < 1e-9, "coverage {covered}");
        let nothing: Vec<&[f64]> = Vec::new();
        assert_eq!(archive.coverage(nothing), 0.0);
    }

    #[test]
    fn the_posterior_follows_what_worked() {
        let mut bandit = Curiosity::new(3);
        // A uniform prior is an even rate and no draws behind it.
        assert!((bandit.score(0) - 0.5).abs() < 1e-12);
        assert_eq!(bandit.draws(0), 0.0);
        bandit.reward(1);
        bandit.reward(1);
        bandit.penalise(2);
        assert!(bandit.score(1) > bandit.score(0));
        assert!(bandit.score(2) < bandit.score(0));
        assert_eq!(bandit.draws(1), 2.0);
    }

    #[test]
    fn a_failing_cell_is_never_written_off() {
        let mut bandit = Curiosity::new(1);
        for _ in 0..200 {
            bandit.penalise(0);
        }
        // A Beta posterior cannot reach zero, which is what stops a
        // descriptor that is wrong about a cell from excluding the
        // region behind it for good.
        assert!(bandit.score(0) > 0.0);
    }

    #[test]
    fn thompson_sampling_prefers_the_cell_that_pays() {
        let mut bandit = Curiosity::new(2);
        for _ in 0..30 {
            bandit.reward(0);
            bandit.penalise(1);
        }
        let mut rng = StdRng::seed_from_u64(13);
        let picked = (0..300)
            .filter(|_| bandit.select(&[0, 1], &mut rng) == Some(0))
            .count();
        assert!(picked > 270, "paying cell chosen only {picked} of 300");
    }

    #[test]
    fn an_untried_cell_still_wins_draws() {
        // The property a decay heuristic does not have: a cell with no
        // evidence keeps a wide posterior and keeps being explored, so
        // effort leaves it only once there is evidence to leave on.
        //
        // The rate is not a guess. Against Beta(13,1) an untried
        // Beta(1,1) is uniform, so it wins with probability
        // \(\int_0^1 13x^{12}(1-x)\,dx = 1/14\), about seven per
        // cent, which is 28 of 400 with a standard deviation near 5.
        // Ten is three standard deviations below that and still well
        // clear of the 0.95 per cent a once-penalised cell would give.
        let mut bandit = Curiosity::new(2);
        for _ in 0..12 {
            bandit.reward(0);
        }
        let mut rng = StdRng::seed_from_u64(19);
        let explored = (0..400)
            .filter(|_| bandit.select(&[0, 1], &mut rng) == Some(1))
            .count();
        assert!(
            explored > 10,
            "untried cell explored only {explored} of 400, against 28 expected"
        );
    }

    #[test]
    fn selection_answers_even_with_nothing_to_go_on() {
        let bandit = Curiosity::default();
        let mut rng = StdRng::seed_from_u64(17);
        assert_eq!(bandit.select(&[], &mut rng), None);
        // Cells outside the table carry the uniform prior; still an
        // answer, and still a fair one.
        assert_eq!(bandit.select(&[4], &mut rng), Some(4));
    }

    #[test]
    fn a_beta_draw_stays_in_the_unit_interval() {
        let mut rng = StdRng::seed_from_u64(23);
        for (alpha, beta) in [(0.5, 0.5), (1.0, 1.0), (30.0, 2.0), (2.0, 30.0)] {
            for _ in 0..200 {
                let drawn = sample_beta(alpha, beta, &mut rng);
                assert!(
                    (0.0..=1.0).contains(&drawn),
                    "Beta({alpha},{beta}) drew {drawn}"
                );
            }
        }
        // A shape the method cannot use answers zero rather than
        // looping or returning a NaN into a comparison.
        assert_eq!(sample_gamma(0.0, &mut rng), 0.0);
        assert_eq!(sample_gamma(f64::NAN, &mut rng), 0.0);
    }

    #[test]
    fn a_lopsided_posterior_concentrates() {
        let mut rng = StdRng::seed_from_u64(29);
        let draws: Vec<f64> = (0..500).map(|_| sample_beta(60.0, 2.0, &mut rng)).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        // Beta(60,2) has mean 60/62; the sampler should find it.
        assert!((mean - 60.0 / 62.0).abs() < 0.02, "mean {mean}");
    }

    #[test]
    fn nothing_seen_is_maximally_novel() {
        assert_eq!(novelty(&[1.0, 1.0], &[], 3), f64::INFINITY);
        assert_eq!(novelty(&[1.0], &[vec![0.0]], 0), 0.0);
    }

    #[test]
    fn novelty_grows_with_distance_from_the_seen() {
        let seen = vec![vec![0.0, 0.0], vec![0.1, 0.0]];
        let near = novelty(&[0.05, 0.0], &seen, 2);
        let far = novelty(&[9.0, 0.0], &seen, 2);
        assert!(far > near, "far {far} near {near}");
    }
}
