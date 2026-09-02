use anneal_core::minima_db::{MinimaCorpus, MinimaSet, MinimaUnits};
use ndarray::Array1;

fn units() -> MinimaUnits {
    MinimaUnits {
        length: "sigma".into(),
        energy: "epsilon".into(),
    }
}

fn set(seed: u64) -> MinimaSet {
    MinimaSet {
        system: "lj13".into(),
        temperature: 0.8,
        seed,
    }
}

#[test]
fn minima_round_trip_exactly_and_fold_across_seeds() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = MinimaCorpus::open(dir.path().join("minima")).unwrap();
    let a = Array1::from(vec![0.0, 0.0, 0.0, 1.122462048309373, 0.0, 0.0]);
    let b = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.122462048309373, 0.1]);
    let c = Array1::from(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let appended = corpus
        .record(
            &set(1),
            &[],
            &units(),
            &[
                (-1.0, a.view()),
                (-1.0 + 1e-9, a.view()),
                (-1.0, c.view()),
                (-0.75, b.view()),
            ],
            1e-6,
            serde_json::json!({"driver": "test"}),
        )
        .unwrap();
    assert_eq!(
        appended, 3,
        "equal energies do not merge structurally different coordinates"
    );
    let appended = corpus
        .record(
            &set(2),
            &[],
            &units(),
            &[(-1.0, a.view()), (-0.5, b.view())],
            1e-6,
            serde_json::json!({"driver": "test"}),
        )
        .unwrap();
    assert_eq!(appended, 2);

    let stored = corpus.minima("lj13", 0.8).unwrap();
    assert_eq!(stored.len(), 5);
    assert_eq!(stored[0].energy, -1.0);
    assert_eq!(
        stored[0].coordinates,
        a.to_vec(),
        "coordinates come back bit for bit"
    );
    assert!(stored.iter().any(|m| m.set.seed == 2 && m.energy == -0.5));
    assert_eq!(
        corpus.distinct_energies("lj13", 0.8, 1e-6).unwrap(),
        vec![-1.0, -0.75, -0.5]
    );
    assert!(
        corpus.minima("lj13", 0.9).unwrap().is_empty(),
        "another temperature is another set"
    );
    assert!(
        corpus.minima("lj38", 0.8).unwrap().is_empty(),
        "another system is another set"
    );

    let reopened = MinimaCorpus::open(dir.path().join("minima")).unwrap();
    assert_eq!(reopened.minima("lj13", 0.8).unwrap().len(), 5);
}

#[test]
fn a_nonfinite_energy_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = MinimaCorpus::open(dir.path().join("minima")).unwrap();
    let a = Array1::from(vec![0.0, 0.0, 0.0]);
    assert!(
        corpus
            .record(
                &set(1),
                &[],
                &units(),
                &[(f64::NAN, a.view())],
                1e-6,
                serde_json::json!({})
            )
            .is_err()
    );
}
