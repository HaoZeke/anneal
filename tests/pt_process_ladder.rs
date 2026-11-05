//! Process workers on [`ParallelTemperingSampler::run_processes`] match `run`.

use anneal_core::cool::LogCool;
use anneal_core::potentials::PairPotential;
use anneal_core::variant::boltzmann;
use anneal_core::{ParallelTemperingSampler, PtResult, geometric_ladder};

fn run_pair(workers: usize, seed: u64, bin: Option<&std::path::Path>) -> (PtResult, PtResult) {
    let chains = 2 * workers;
    let potential = PairPotential::lennard_jones(4);
    let sampler = boltzmann(potential, 0.8, 0.38).expect("variant");
    let temps = geometric_ladder(0.05, 0.8, chains);
    let tempering = ParallelTemperingSampler::new(sampler, temps, 23, 5);
    let cooling = LogCool::new(0.8, 2.0);
    let serial = tempering.run(&cooling, 2, seed);
    let parallel = tempering.run_processes(&cooling, 2, seed, workers, bin);
    (serial, parallel)
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
fn library_process_workers_match_the_serial_ladder() {
    let bin = env!("CARGO_BIN_EXE_weak_scale");
    for (workers, seed) in [(2usize, 11u64), (3, 4), (4, 19)] {
        let (serial, parallel) = run_pair(workers, seed, Some(std::path::Path::new(bin)));
        assert_same_ladder(&serial, &parallel);
    }
}
