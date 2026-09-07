use anneal_core::allocate::{DepthAllocator, RewardMoments};
use anneal_core::surface_evidence::{SurfaceEvidenceBook, SurfaceReport};

fn moments(values: &[f64]) -> RewardMoments {
    let mut moments = RewardMoments::default();
    for &value in values {
        moments.observe(value).unwrap();
    }
    moments
}

fn report(schema: &str, arms: &[&[f64]]) -> SurfaceReport {
    SurfaceReport {
        schema: schema.to_owned(),
        arms: arms.iter().map(|values| moments(values)).collect(),
    }
}

#[test]
fn cumulative_reports_share_peer_evidence_without_echo_or_replay_credit() {
    let mut book = SurfaceEvidenceBook::default();
    let teacher = report("surface-depth-v1/block=100", &[&[-2.0, -4.0], &[3.0]]);
    let learner = report("surface-depth-v1/block=100", &[&[], &[]]);
    let own_reply = book.exchange(0, teacher.clone()).unwrap();
    assert_eq!(own_reply.arms, learner.arms);
    assert_eq!(book.exchange(1, learner.clone()).unwrap(), teacher);
    assert_eq!(book.exchange(0, teacher.clone()).unwrap(), own_reply);
    assert_eq!(book.exchange(1, learner).unwrap(), teacher);
}

#[test]
fn incremental_reports_replace_cumulative_evidence_instead_of_adding_it() {
    let mut book = SurfaceEvidenceBook::default();
    book.exchange(0, report("schema", &[&[1.0], &[-2.0]])).unwrap();
    let updated = report("schema", &[&[1.0, 3.0], &[-2.0, -4.0]]);
    book.exchange(0, updated.clone()).unwrap();
    let peer = book.exchange(1, report("schema", &[&[], &[]])).unwrap();
    assert_eq!(peer, updated);
}

#[test]
fn incompatible_surface_experiments_do_not_share_rewards() {
    let mut book = SurfaceEvidenceBook::default();
    book.exchange(0, report("block=100", &[&[5.0]])).unwrap();
    let other = report("block=500", &[&[]]);
    assert_eq!(book.exchange(1, other.clone()).unwrap(), other);
}

#[test]
fn invalid_or_regressing_reports_leave_the_shared_evidence_unchanged() {
    let mut book = SurfaceEvidenceBook::default();
    let valid = report("schema", &[&[1.0, 3.0]]);
    book.exchange(0, valid.clone()).unwrap();
    for invalid in [
        report("schema", &[&[1.0]]),
        report("schema", &[&[2.0, 3.0]]),
        SurfaceReport { schema: "schema".into(), arms: vec![RewardMoments { count: 3, mean: f64::NAN, m2: 0.0 }] },
        SurfaceReport { schema: "schema".into(), arms: vec![RewardMoments { count: 3, mean: 0.0, m2: -1.0 }] },
    ] {
        assert!(book.exchange(0, invalid).is_err());
        assert_eq!(book.exchange(1, report("schema", &[&[]])).unwrap(), valid);
    }
}

#[test]
fn merged_moments_reproduce_sequential_normal_gamma_updates() {
    let observations = [-4.0, 2.0, 1.0, -3.0, 8.0];
    let pooled = moments(&observations[..2]).merge(moments(&observations[2..])).unwrap();
    let direct = moments(&observations);
    assert_eq!(pooled.count, direct.count);
    assert!((pooled.mean - direct.mean).abs() < 1e-12);
    assert!((pooled.m2 - direct.m2).abs() < 1e-12);
    let reconstructed = DepthAllocator::from_moments(&[pooled]).unwrap();
    let mut sequential = DepthAllocator::new(1);
    for reward in observations { sequential.update(0, reward); }
    assert_eq!(reconstructed.draws, sequential.draws);
    assert!((reconstructed.means()[0] - sequential.means()[0]).abs() < 1e-12);
    use rand::{SeedableRng, rngs::StdRng};
    let mut left = StdRng::seed_from_u64(79);
    let mut right = left.clone();
    for _ in 0..64 {
        assert_eq!(reconstructed.select(&mut left), sequential.select(&mut right));
    }
}
