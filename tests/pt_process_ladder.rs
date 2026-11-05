//! Process workers match `ParallelTemperingSampler::run` on the Lennard-Jones ladder.

use anneal_core::PtResult;
use anneal_core::weak_scale::{LjLadder, lj_ladder_result, serial_lj_pt};

fn tiny(workers: usize, seed: u64) -> LjLadder {
    LjLadder {
        atoms: 4,
        chains_per_worker: 2,
        steps: 23,
        swap_period: 5,
        epochs: 2,
        workers,
        seed,
    }
}

fn assert_same_ladder(serial: &PtResult, parallel: &PtResult) {
    assert_eq!(serial.best_val.to_bits(), parallel.best_val.to_bits());
    assert_eq!(serial.best_pos.len(), parallel.best_pos.len());
    for (left, right) in serial.best_pos.iter().zip(parallel.best_pos.iter()) {
        assert_eq!(left.to_bits(), right.to_bits());
    }
    assert_eq!(serial.swap_attempts, parallel.swap_attempts);
    assert_eq!(serial.swap_accepts, parallel.swap_accepts);
    assert_eq!(
        serial.log_z_ratio_est.to_bits(),
        parallel.log_z_ratio_est.to_bits()
    );
    assert_eq!(
        serial.log_z_ratio_se.to_bits(),
        parallel.log_z_ratio_se.to_bits()
    );
    assert_eq!(serial.chain_histories.len(), parallel.chain_histories.len());
    for (left, right) in serial
        .chain_histories
        .iter()
        .zip(parallel.chain_histories.iter())
    {
        assert_eq!(left.best.val.to_bits(), right.best.val.to_bits());
        assert_eq!(left.best.pos.len(), right.best.pos.len());
        for (left_pos, right_pos) in left.best.pos.iter().zip(right.best.pos.iter()) {
            assert_eq!(left_pos.to_bits(), right_pos.to_bits());
        }
        assert_eq!(left.epochs.len(), right.epochs.len());
        for (epoch_left, epoch_right) in left.epochs.iter().zip(right.epochs.iter()) {
            assert_eq!(epoch_left.accepted, epoch_right.accepted);
            assert_eq!(epoch_left.rejected, epoch_right.rejected);
            assert_eq!(
                epoch_left.best_val.to_bits(),
                epoch_right.best_val.to_bits()
            );
            assert_eq!(epoch_left.temp.to_bits(), epoch_right.temp.to_bits());
        }
        assert_eq!(left.stationarity_flags, right.stationarity_flags);
    }
}

#[test]
fn process_workers_match_the_serial_ladder() {
    let bin = env!("CARGO_BIN_EXE_weak_scale");
    for (workers, seed) in [(2usize, 11u64), (3, 4), (4, 19)] {
        let spec = tiny(workers, seed);
        let serial = serial_lj_pt(&spec);
        let parallel = lj_ladder_result(&spec, Some(std::path::Path::new(bin)));
        assert_same_ladder(&serial, &parallel);
    }
}
