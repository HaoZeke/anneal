//! Occupancy extras Leave OtherFamily or ArchiveHole. A same-family
//! quench is Refuse, then a packing hole; it is not a box start.

#[test]
fn catalog_min_families_override_requires_a_parsed_floor() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let floor = source
        .split("fn occupancy_floor(")
        .nth(1)
        .expect("occupancy_floor must exist");
    let body = floor
        .split("fn leftover_census_dwell(")
        .next()
        .expect("floor ends at leftover dwell");
    assert!(
        body.contains("parse()"),
        "an empty CATALOG_MIN_FAMILIES must not skip the Fiedler floor"
    );
    assert!(
        !body.contains("is_ok()"),
        "presence of CATALOG_MIN_FAMILIES is not a paper-floor override"
    );
    assert!(
        body.contains("occupancy_landfold_from_book"),
        "the book landfold community count is the secondary family floor"
    );
    assert!(
        body.contains("sparsified_book"),
        "peeled book communities drive Leave F"
    );
    assert!(
        !body.contains("occupancy_ring_from_book"),
        "Franzblau ring class is a Leave lens, not a retire floor"
    );
}

#[test]
fn packing_census_uses_the_sparsified_book() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let body = source
        .split("fn packing_census_saturated(")
        .nth(1)
        .expect("packing_census_saturated must exist")
        .split("fn occupancy_funnel_ei_exhausted(")
        .next()
        .expect("packing census ends at funnel EI");
    assert!(
        body.contains("sparsified_book"),
        "packing saturation is Chao1 on the landfold-sparsified book"
    );
    assert!(
        body.contains(".holes"),
        "Leave continues only while the sparsified book has holes"
    );
}

#[test]
fn occupancy_report_emits_the_landfold_book_map() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let report = source
        .split("fn report_occupancy_gt(")
        .nth(1)
        .expect("occupancy report must exist")
        .split("fn record_energy(")
        .next()
        .expect("occupancy report must end at energy recording");
    assert!(
        report.contains("sparsified.saturated()"),
        "reported packing_sat is the sparsified book, not leftover n1 Chao1"
    );
    assert!(
        !report.contains("packing.chao1_complete()"),
        "leftover well singletons are not the packing stop"
    );
    assert!(
        report.contains("report_occupancy_landfold"),
        "the book landfold map must be written for the figure path"
    );
    assert!(
        report.contains("landfold_holes"),
        "occupancy_gt must report whether the sparsified book still has holes"
    );
    assert!(
        report.contains("\\\"kind\\\":\\\"occupancy_landfold\\\""),
        "plot records are occupancy_landfold JSONL"
    );
}

#[test]
fn occupancy_fes_report_uses_independent_well_arrivals_only() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let report = source
        .split("fn report_occupancy_gt(")
        .nth(1)
        .expect("occupancy report must exist")
        .split("fn record_energy(")
        .next()
        .expect("occupancy report must end at energy recording");
    assert!(
        report.contains("occupancy_fes_delta(&scientific.packing.occupied_well_counts())"),
        "packing FES must use independent leftover-well counts"
    );
    assert!(
        !report.contains("occupancy_fes_from_wells"),
        "last, best, and catalog containers are not occupancy samples"
    );
    assert!(report.contains("\\\"fes_delta\\\""));
    assert!(
        report.contains("sparsified.fes_minima"),
        "fes_minima is the book-map FES, not last/best containers"
    );
    assert!(report.contains("\\\"fes_minima\\\""));
    assert!(!report.contains("\\\"fes_map_delta\\\""));
}

#[test]
fn occupancy_fes_report_key_tracks_the_discrete_gap() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let report = source
        .split("fn report_occupancy_gt(")
        .nth(1)
        .expect("occupancy report must exist")
        .split("fn record_energy(")
        .next()
        .expect("occupancy report must end at energy recording");
    let key = report
        .split("let key = OccupancyGtKey {")
        .nth(1)
        .expect("occupancy report suppression key must exist")
        .split("};")
        .next()
        .expect("occupancy report suppression key must close");
    assert!(
        key.contains("fes_delta_bits: fes_delta.map(f64::to_bits)"),
        "a changed FES gap must emit a new occupancy report"
    );
}

#[test]
fn leftover_census_dwell_requires_the_sat_streak() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let dwell = source
        .split("fn leftover_census_dwell(")
        .nth(1)
        .expect("leftover_census_dwell must exist")
        .split("fn report_occupancy_gt(")
        .next()
        .expect("leftover dwell ends at occupancy report");
    assert!(
        dwell.contains("leftover_dwell_from_census"),
        "live leftover_dwell is leftover_dwell_from_census, not a one-shot nick"
    );
    assert!(
        dwell.contains("leftover_sat_streak"),
        "leftover_dwell must require the consecutive leftover-sat streak"
    );
}

#[test]
fn live_leave_gate_does_not_document_the_4000_hop_floor() {
    let hop = include_str!("../examples/lj_cluster_search.rs");
    assert!(
        !hop.contains("First 4000 hops stay on the walk"),
        "live Leave comment must name LEAVE_CROSSING_HOPS, not the expired 4000-hop floor"
    );
    assert!(
        hop.contains("measured crossing floor (LEAVE_CROSSING_HOPS)"),
        "live Leave comment must name the crossing-floor constant"
    );
    let occupancy = include_str!("../src/catalog/occupancy.rs");
    let dwell_docs = occupancy
        .split("pub const LEFTOVER_SAT_DWELL")
        .next()
        .expect("LEFTOVER_SAT_DWELL must exist")
        .rsplit("/// Consecutive leftover-sat")
        .next()
        .expect("LEFTOVER_SAT_DWELL docs must start at the consecutive-sat sentence");
    assert!(
        dwell_docs.contains("leftover_dwell_from_census"),
        "LEFTOVER_SAT_DWELL docs name leftover_dwell_from_census as live dwell"
    );
    assert!(
        !dwell_docs.contains("leftover_hatch_stable"),
        "LEFTOVER_SAT_DWELL is the census streak, not leftover_hatch_stable"
    );
}

#[test]
fn occupancy_fes_report_does_not_advance_leftover_dwell() {
    let source = include_str!("../src/catalog_rpc/server.rs");
    let report = source
        .split("fn report_occupancy_gt(")
        .nth(1)
        .expect("occupancy report must exist")
        .split("fn record_energy(")
        .next()
        .expect("occupancy report must end at energy recording");
    assert!(
        !report.contains("leftover_sat_streak"),
        "diagnostic report changes must not advance the retirement dwell"
    );
}

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
        .skip(1)
        .find(|chunk| chunk.contains("occupancy_leave_by_birth"))
        .expect("occupancy extras Leave arm must exist");
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
    assert!(
        arm.contains("occupied_family_count"),
        "OtherFamily is landfold communities on the sparsified book, not DECAF isomer bins"
    );
    assert!(
        arm.contains("leave_archive_hole") || arm.contains("leave_packing_state"),
        "ArchiveHole is leftover-orthogonal, not a leftover hole"
    );
    let server = include_str!("../src/catalog_rpc/server.rs");
    assert!(
        server.contains("q_ei_family_entry"),
        "WAVE OtherFamily draws cycle q-EI, not a single highest-EI family"
    );
    assert!(
        arm.contains("archive_cover_index") || arm.contains("cover_index"),
        "ArchiveHole is a SoftSaddle covering direction, not a random nu3 kick"
    );
    assert!(
        !arm.contains("step_away_fivefold"),
        "ArchiveHole is not the serial fivefold morphology hop"
    );
}

#[test]
fn leave_quench_keeps_the_walk_off_mu_k() {
    let source = include_str!("../src/methods/cluster_hopping.rs");
    let body = source
        .split("let leave_action = crate::catalog::is_occupancy_leave_action")
        .nth(1)
        .expect("Leave quench arm must exist")
        .split("if leave == Some(crate::catalog::OccupancyLeaveAdopt::HoleStep)")
        .next()
        .expect("Leave quench arm ends at HoleStep");
    assert!(
        body.contains("leaves_packing"),
        "Leave adopt is the packing community, not a rise in packing-mean span"
    );
    assert!(
        !body.contains("occupancy_leave_new_class"),
        "the cell grain is not the Leave polish or adopt bit"
    );
    assert!(
        !body.contains("leave_occupied_packing"),
        "leftover-SOAP requench is a projector onto the occupied packing"
    );
    assert!(
        body.contains("leave_packing_starts") || body.contains("leave_packing_ladder"),
        "a Leave that quenched back into its packing widens the ladder"
    );
    assert!(
        body.contains("leave_packing_rung_to"),
        "a rung is sized by the barrier it must clear, measured on the potential"
    );
    assert!(
        body.contains("leave_packing_ridge"),
        "a Leave that the ladder refuses accumulates the packing increment"
    );
    // A min-mode climb on the raw surface is the surface rumple: a closed
    // shell has no soft mode, the climb reports curvatures of -1e13 and
    // calls the ridge behind after one step. On E+V the shell is not
    // closed, the deposit having put fifty eps into it, and the same climb
    // measures a curvature of 0.5218 and leaves the packing 3 times in 16
    // where every displacement along a cover direction leaves 0 in 16. So
    // the bar is not that no climb appears, it is that no climb runs on
    // the raw surface.
    if body.contains("activate_from_origin") {
        let climb = body
            .split("activate_from_origin")
            .next()
            .expect("text precedes the climb");
        assert!(
            climb.contains("with_hill_only"),
            "a Leave climb runs on E+V; the min mode of a closed shell is a surface rumple"
        );
    }
    assert!(
        !body.contains("with_hill_only") || body.contains("with_disarmed"),
        "a climb on E+V still needs the raw minimum below it to name the packing"
    );
    assert!(
        body.contains("LEAVE_WALK_CLIMB"),
        "a crossing above the climb ceiling is a crushed cluster, not a saddle"
    );
    assert!(
        body.contains("with_disarmed"),
        "the walk stops on a ridge, so the chain takes the raw minimum below it"
    );
}

#[test]
fn a_one_packing_book_walks_rather_than_drawing_a_hole() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let leave = source
        .split("PolicyAction::Leave =>")
        .skip(1)
        .find(|chunk| chunk.contains("occupancy_leave_by_birth"))
        .expect("occupancy extras Leave arm must exist");
    let arm = leave
        .split("PolicyAction::Explore =>")
        .next()
        .expect("Leave arm must end at Explore");
    assert!(
        arm.contains("occupancy_leave_by_birth"),
        "Walk vs ArchiveHole follows FunnelModel EI plus leftover birth, not only the community count"
    );
    assert!(
        arm.contains("OccupancyLeaveTarget::Walk"),
        "a book with one packing has nothing to divide, so the extra keeps walking"
    );
    let walk = arm
        .split("OccupancyLeaveTarget::Walk =>")
        .nth(1)
        .expect("Walk arm must exist")
        .split("OccupancyLeaveTarget::OtherFamily =>")
        .next()
        .expect("Walk arm ends at OtherFamily");
    assert!(
        walk.contains("CheckpointAction::Continue"),
        "Walk keeps the replica on its own trajectory"
    );
    assert!(
        !walk.contains("leave_packing_state"),
        "Walk does not draw a hole: measured on LJ75, no rung from 1.32 to 42.3 eps leaves the packing"
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

#[test]
fn putative_saturated_does_not_latch_the_done_line() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let retire = source
        .split("if occupancy_retire_at(")
        .nth(1)
        .expect("retire latch must exist");
    let after_retire = retire
        .split("let policy_trace = cooperative")
        .next()
        .expect("retire block must end at the policy trace");
    assert!(
        after_retire.contains("announced_putative"),
        "CatalogSaturated putative must not share the done latch"
    );
    assert!(
        !after_retire
            .split("return CheckpointAction::Retire")
            .nth(1)
            .expect("putative print follows Retire")
            .contains("announced_done = true"),
        "putative saturated must not suppress a later done mixing line"
    );
}

#[test]
fn slice_diagnostics_report_the_validated_local_best() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let checkpoint = source
        .split("let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {")
        .nth(1)
        .expect("catalog checkpoint callback must exist")
        .split("let outcome = run_with_bias_at_checkpoints")
        .next()
        .expect("catalog checkpoint callback must end at the run");
    let compact: String = checkpoint.split_whitespace().collect();
    assert!(
        compact.contains("energy:Some(snapshot.best_energy())"),
        "slice diagnostics must expose the validated local best"
    );
    assert!(
        !compact.contains("energy:Some(snapshot.current_energy())"),
        "leaving a target basin within one slice must not erase its encounter"
    );
    assert!(
        !compact.contains("energy:Some(parent.energy)")
            && !compact.contains("trace.energy=Some(candidate.energy)"),
        "an unvalidated remote candidate is not a local target encounter"
    );
    assert!(
        !compact.contains("checkpoint_charged,snapshot.current_energy(),"),
        "early checkpoint exits must retain the validated local best"
    );
}

#[test]
fn foreign_parent_population_reseed_is_not_a_box_start() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let extra = source
        .split("if foreign_parent {")
        .nth(1)
        .expect("foreign-parent population reseed arm must exist");
    let arm = extra
        .split("slice_sequence = slice_sequence")
        .next()
        .expect("foreign-parent arm must end at the slice record");
    assert!(
        !arm.contains("random_cluster"),
        "Feynman-Kac extras Leave a SOAP hole or packing kick, not a box start"
    );
}
