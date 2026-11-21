fn scan_safe6_gate_report(
    outcome: &anneal_core::methods::cluster_hopping::Outcome,
    wall: f64,
    potential: f64,
) {
    use sha2::Digest as _;
    use std::io::Write as _;

    let Ok(path) = std::env::var("SCAN_SAFE6_TRACE") else {
        return;
    };
    let state_hash = |state: Option<&ndarray::Array1<f64>>| {
        let mut digest = sha2::Sha256::new();
        match state {
            Some(values) => {
                digest.update([1]);
                for value in values {
                    digest.update(value.to_bits().to_le_bytes());
                }
            }
            None => digest.update([0]),
        }
        format!("{:x}", digest.finalize())
    };
    let mut trajectory = sha2::Sha256::new();
    for edge in &outcome.accepted_transitions {
        trajectory.update((edge.hop as u64).to_le_bytes());
        trajectory.update((edge.action.len() as u64).to_le_bytes());
        trajectory.update(edge.action.as_bytes());
        trajectory.update(edge.from_energy.to_bits().to_le_bytes());
        trajectory.update(edge.to_energy.to_bits().to_le_bytes());
        trajectory.update([u8::from(edge.adopted), u8::from(edge.validated)]);
        for value in &edge.to_state {
            trajectory.update(value.to_bits().to_le_bytes());
        }
    }
    let trajectory_hash = format!("{:x}", trajectory.finalize());
    let final_hash = state_hash(outcome.final_state.as_ref());
    let file = std::fs::File::create(path).expect("create gate trace");
    let mut out = std::io::BufWriter::new(file);
    serde_json::to_writer(
        &mut out,
        &serde_json::json!({
            "kind": "summary",
            "wall": wall,
            "potential": potential,
            "calls": outcome.charged,
            "energy": outcome.final_energy,
            "energy_bits": format!("{:016x}", outcome.final_energy.to_bits()),
            "hash": final_hash,
            "trajectory": trajectory_hash,
        }),
    )
    .expect("write gate summary");
    writeln!(out).expect("finish gate summary");
    for (index, edge) in outcome.accepted_transitions.iter().enumerate() {
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({
                "kind": "transition",
                "index": index,
                "step": edge.hop,
                "action": edge.action,
                "from_bits": format!("{:016x}", edge.from_energy.to_bits()),
                "to_bits": format!("{:016x}", edge.to_energy.to_bits()),
                "adopted": edge.adopted,
                "validated": edge.validated,
                "hash": state_hash(Some(&edge.to_state)),
            }),
        )
        .expect("write gate transition");
        writeln!(out).expect("finish gate transition");
    }
    for (index, (step, calls, basins, energy)) in
        outcome.improvements.iter().enumerate()
    {
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({
                "kind": "improvement",
                "index": index,
                "step": step,
                "call": calls,
                "basins": basins,
                "energy_bits": format!("{:016x}", energy.to_bits()),
            }),
        )
        .expect("write gate improvement");
        writeln!(out).expect("finish gate improvement");
    }
    out.flush().expect("flush gate trace");
}
