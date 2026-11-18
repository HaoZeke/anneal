use anneal_core::catalog::{SeamBank, local_packing_cohesion, packing_doorway_channel};
use anneal_core::methods::cluster_hopping::Config;
use ndarray::Array1;

#[test]
fn the_measured_shares_fire_inside_two_epsilon_and_a_doorway_does_not() {
    let floor = -173.928427;
    assert!(
        packing_doorway_channel(floor + 2.0, floor, 0.120, 0.307),
        "a reduced packing cohesion on the floor is the channel"
    );
    assert!(
        !packing_doorway_channel(floor + 2.0 + 1e-9, floor, 0.120, 0.307),
        "two epsilon is the window"
    );
    assert!(
        !packing_doorway_channel(floor, floor, 0.307, 0.307),
        "an uncollapsed cohesion is not a fire"
    );
    assert!(
        !packing_doorway_channel(floor + 6.0, floor, 0.120, 0.307),
        "the twelve-epsilon doorway is not the channel"
    );
    assert!(!packing_doorway_channel(floor, f64::INFINITY, 0.120, 0.307));
    assert!(SeamBank::bin_of(0.45).is_some());
    assert!(!packing_doorway_channel(floor + 1.86, floor, 0.307, 0.307));
    let load_xyz = |text: &str| -> Array1<f64> {
        Array1::from_vec(
            text.lines()
                .skip(2)
                .filter(|line| !line.trim().is_empty())
                .flat_map(|line| {
                    line.split_whitespace()
                        .skip(1)
                        .take(3)
                        .map(|v| v.parse::<f64>().unwrap())
                })
                .collect(),
        )
    };
    let regular_tetra = Array1::from_vec(vec![
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
    ]);
    let symmetric_cohesion = local_packing_cohesion(regular_tetra.view());
    assert!(
        (symmetric_cohesion - 1.0).abs() < 1e-12,
        "a transitive vertex orbit must have unit Simpson cohesion, got {symmetric_cohesion}"
    );
    let fcc38 = load_xyz(include_str!("fixtures/lj38_fcc.xyz"));
    let mut doorway38 = fcc38.clone();
    doorway38[0] += 0.85;
    doorway38[4] -= 0.75;
    doorway38[8] += 0.65;
    let fcc_cohesion = local_packing_cohesion(fcc38.view());
    let doorway_cohesion = local_packing_cohesion(doorway38.view());
    assert!(
        doorway_cohesion < fcc_cohesion && doorway_cohesion > 0.0,
        "a surface-displaced doorway must strictly lower local packing cohesion: doorway {doorway_cohesion} vs floor {fcc_cohesion}"
    );
    assert!(packing_doorway_channel(
        floor + 1.5,
        floor,
        doorway_cohesion,
        fcc_cohesion,
    ));
    assert!(!packing_doorway_channel(
        floor + 1.5,
        floor,
        fcc_cohesion,
        doorway_cohesion,
    ));
}

#[test]
fn restart_is_not_a_recommended_stall_arm() {
    for n in [38usize, 75, 98] {
        assert!(!Config::recommended(n).restart_on_stall);
        assert!(!Config::derived(n).restart_on_stall);
        assert!(Config::recommended(n).escape_on_stall);
        assert!(!Config::communicating(n).restart_on_stall);
        assert!(!Config::communicating(n).escape_on_stall);
    }
}
