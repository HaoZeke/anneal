//! The paper seed registry is committed and names every registered run.

#[test]
fn the_registry_lists_the_paper_seed_ranges() {
    let text = include_str!("../paper/seeds.toml");
    let table: toml::Value = toml::from_str(text).expect("paper/seeds.toml parses");
    let runs = table
        .get("run")
        .and_then(toml::Value::as_array)
        .expect("[[run]] entries");
    let range = |id: &str| -> (i64, i64) {
        let run = runs
            .iter()
            .find(|run| run.get("id").and_then(toml::Value::as_str) == Some(id))
            .unwrap_or_else(|| panic!("missing run {id}"));
        let start = run.get("seed_start").and_then(toml::Value::as_integer);
        let end = run.get("seed_end").and_then(toml::Value::as_integer);
        (start.expect("seed_start"), end.expect("seed_end"))
    };
    assert_eq!(range("r1-recommended-vs-sbh"), (2000, 2199));
    assert_eq!(range("r12-derived"), (2000, 2199));
    assert_eq!(range("r15-gmin"), (2000, 2199));
    assert_eq!(range("r2-held-out"), (2200, 2247));
    assert_eq!(range("r4-lattice-lj75-lj98"), (3000, 3029));
    assert_eq!(range("r4-lattice-lj38"), (3100, 3199));
    assert_eq!(range("r9-pt111-heptamer"), (4000, 4047));
    assert_eq!(range("r10-soap"), (5000, 5063));
    assert_eq!(range("r6-water-hexamer"), (0, 31));
    assert_eq!(runs.len(), 9);
}
