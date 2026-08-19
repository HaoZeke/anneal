//! Occupancy extras Leave OtherFamily or ArchiveHole. A same-family
//! quench is Refuse, then a packing hole; it is not a box start.

#[test]
fn occupancy_leave_refuse_is_not_a_box_start() {
    let source = include_str!("../src/methods/cluster_hopping.rs");
    let after = source
        .split("if leave == Some(crate::catalog::OccupancyLeaveAdopt::Refuse)")
        .nth(1)
        .expect("refuse arm must exist");
    let arm = after
        .split("if leave == Some(crate::catalog::OccupancyLeaveAdopt::HoleStep)")
        .next()
        .expect("refuse arm must end at HoleStep");
    assert!(
        !arm.contains("random_cluster"),
        "same-family occupancy Leave must stay or packing-kick, not box-start"
    );
}

#[test]
fn occupancy_leave_action_does_not_fall_back_to_a_random_cluster() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let leave = source
        .split("PolicyAction::Leave =>")
        .nth(1)
        .expect("Leave arm must exist");
    let arm = leave
        .split("PolicyAction::Explore =>")
        .next()
        .expect("Leave arm must end at Explore");
    assert!(
        !arm.contains("random_cluster"),
        "occupancy extras Leave OtherFamily or ArchiveHole, not a random cluster"
    );
    assert!(
        arm.contains("packing_saturated") || arm.contains("policy.packing_saturated"),
        "after packing sat Leave must see packing_saturated and choose ArchiveHole"
    );
}

#[test]
fn occupied_packing_extras_do_not_reseed_a_random_cluster() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let extra = source
        .split("extra_of_occupied_packing")
        .nth(2)
        .expect("extra-of-occupied-packing arm must exist");
    let arm = extra
        .split("slice_sequence = slice_sequence")
        .next()
        .expect("extra arm must end at the slice record");
    assert!(
        !arm.contains("random_cluster"),
        "occupied-packing extras Leave a hole, they do not box-start"
    );
}
