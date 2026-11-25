// Copyright (c) anneal developers
//! Au55 cooperative arm: four chains and one out-of-process catalog.
//!
//! ```text
//! arch_cooperative serve <seed> <budget>
//! arch_cooperative seed <seed> <budget> <revision> <libira-sha256>
//! ```
//!
//! `seed` prints one CSV row. The budget is the ensemble total and must
//! be a positive multiple of four. Without the `bank-rpc` feature the
//! binary reports that the feature is required and exits 2.

fn main() {
    anneal_core::install_rgpot_kernels!();
    let code = match real_main() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("arch_cooperative: {error}");
            error_code(&error)
        }
    };
    std::process::exit(code);
}

#[cfg(not(feature = "bank-rpc"))]
fn real_main() -> Result<(), String> {
    Err("arch_cooperative requires the bank-rpc feature".into())
}

#[cfg(not(feature = "bank-rpc"))]
fn error_code(error: &str) -> i32 {
    if error.contains("bank-rpc") { 2 } else { 1 }
}

#[cfg(feature = "bank-rpc")]
fn real_main() -> Result<(), String> {
    driver::run()
}

#[cfg(feature = "bank-rpc")]
fn error_code(_error: &str) -> i32 {
    1
}

#[cfg(feature = "bank-rpc")]
mod driver {
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use anneal_core::catalog::{
        DescriptorSignature, EngineSignature, FreshEvaluation, SystemSignature, ValidatorConfig,
        euclidean_gradient_norm,
    };
    use anneal_core::catalog_policy::PolicyAction;
    use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
    use anneal_core::catalog_rpc::server::{CatalogServer, ServerConfig};
    use anneal_core::catalog_rpc::{CatalogCandidate, CatalogIdentity, INCUMBENT_SAMPLE_DRAW};
    use anneal_core::cooperative_search::ledger::{ChargeKind, ChargeSummary, CooperativeLedger};
    use anneal_core::cooperative_search::{
        CatalogOfferOutcome, CatalogSampleOutcome, CooperativeRun, PolicyEvidenceOutcome,
        PolicyRole, ProposalFamily, SliceAdoption, SliceQuench, SliceTrace, SliceValidation,
    };
    use anneal_core::descriptor_space::{
        DescriptorGeometry, DescriptorSpace, universal_descriptor_space,
    };
    use anneal_core::methods::cluster_hopping::{
        ChainCheckpoint, CheckpointAction, Config, Ledger, QuenchBoundary, QuenchStatus,
        random_cluster_in_radius, run_with_energy_gradient_at_checkpoints,
    };
    use anneal_core::methods::warm_lbfgs::WarmLbfgs;
    use anneal_core::minima_db::{MinimaCorpus, MinimaSet, MinimaUnits};
    use anneal_core::potentials::{GOLD_Z, Gupta};
    use anneal_core::{DEFAULT_HIT_TOL, energy_hits};
    use ndarray::{Array1, ArrayView1};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use sha2::{Digest, Sha256};

    const N: usize = 55;
    const CHAINS: usize = 4;
    const CHAIN_IDS: [u32; CHAINS] = [0, 1, 2, 3];
    const REFERENCE: f64 = -191.03085280;
    const SLICE: usize = 500;
    const CAMPAIGN: &str = "au55-cooperative";
    const KINDS: [ChargeKind; 11] = [
        ChargeKind::LocalProposal,
        ChargeKind::RemoteProposal,
        ChargeKind::DescriptorEvaluation,
        ChargeKind::AcceptedQuench,
        ChargeKind::RejectedQuench,
        ChargeKind::FreshValidation,
        ChargeKind::Retry,
        ChargeKind::RpcFallback,
        ChargeKind::AuxiliaryEvaluation,
        ChargeKind::BasinEscape,
        ChargeKind::SaddleRide,
    ];

    struct EnsembleClock {
        spent: [AtomicU64; CHAINS],
        hit_at: AtomicU64,
        stop: Arc<AtomicBool>,
    }

    impl EnsembleClock {
        fn new() -> Self {
            Self {
                spent: [
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                ],
                hit_at: AtomicU64::new(u64::MAX),
                stop: Arc::new(AtomicBool::new(false)),
            }
        }

        fn publish(&self, replica: usize, spent: usize) {
            let value = u64::try_from(spent).unwrap_or(u64::MAX);
            self.spent[replica].store(value, Ordering::Release);
        }

        fn note_quench(&self, replica: usize, spent: usize, energy: f64) {
            self.publish(replica, spent);
            if energy_hits(energy, REFERENCE, DEFAULT_HIT_TOL) {
                let total: u64 = self
                    .spent
                    .iter()
                    .map(|slot| slot.load(Ordering::Acquire))
                    .sum();
                self.hit_at.fetch_min(total, Ordering::AcqRel);
                self.stop.store(true, Ordering::Release);
            }
        }
    }

    struct ChainOut {
        replica: u32,
        best: f64,
        best_state: Option<Array1<f64>>,
        charged: usize,
        summaries: [ChargeSummary; 11],
        ensemble_total: u64,
    }

    struct RunningServer {
        child: Child,
    }

    impl Drop for RunningServer {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args().skip(1);
        match args.next().as_deref() {
            Some("serve") => {
                let seed = required_u64(args.next(), "seed")?;
                let budget = required_u64(args.next(), "budget")?;
                reject_extra(args.next(), "serve")?;
                serve(seed, budget)
            }
            Some("seed") => {
                let seed = required_u64(args.next(), "seed")?;
                let budget = required_u64(args.next(), "budget")?;
                let revision = args.next().ok_or("seed takes a revision")?;
                let libira = args.next().ok_or("seed takes a libira sha256")?;
                reject_extra(args.next(), "seed")?;
                run_seed(seed, budget, &revision, &libira)
            }
            _ => Err("usage: arch_cooperative serve|seed".into()),
        }
    }

    fn required_u64(value: Option<String>, name: &str) -> Result<u64, String> {
        let value = value.ok_or_else(|| format!("missing {name}"))?;
        value
            .parse::<u64>()
            .map_err(|_| format!("{name} is not an unsigned integer"))
    }

    fn reject_extra(extra: Option<String>, command: &str) -> Result<(), String> {
        if extra.is_some() {
            Err(format!("{command} received an unexpected argument"))
        } else {
            Ok(())
        }
    }

    fn per_chain_budget(budget: u64) -> Result<u64, String> {
        let chains = u64::try_from(CHAINS).expect("chain count");
        if budget == 0 || budget % chains != 0 {
            return Err("budget must be a positive multiple of the chain count".into());
        }
        Ok(budget / chains)
    }

    fn ensemble_name(seed: u64) -> String {
        format!("au55-{seed}")
    }

    fn chain_seed(seed: u64, replica: u32) -> u64 {
        seed.wrapping_add(u64::from(replica).wrapping_mul(0x9E37_79B9))
    }

    fn gold_digest() -> [u8; 32] {
        let mut hasher = Sha256::new();
        for value in [0.2061_f64, 1.790, 10.229, 4.036, 2.884] {
            hasher.update(value.to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn descriptor_space() -> Result<DescriptorSpace, String> {
        let geometry = DescriptorGeometry::finite(1.0).map_err(|error| error.to_string())?;
        Ok(universal_descriptor_space(geometry))
    }

    fn system_signature(space: &DescriptorSpace) -> SystemSignature {
        let mut hyperparameters = BTreeMap::new();
        hyperparameters.insert("length_unit".to_owned(), "angstrom".to_owned());
        hyperparameters.insert("energy_unit".to_owned(), "eV".to_owned());
        SystemSignature {
            atomic_numbers: vec![GOLD_Z; N],
            coordinate_dim: u64::try_from(N * 3).expect("coordinate dimension"),
            group_labels: (0..u32::try_from(N).expect("atom count")).collect(),
            group_schema: "homogeneous-atoms".to_owned(),
            frozen_mask: vec![false; N],
            cell: None,
            periodic: [false; 3],
            length_scale: 1.0,
            energy_scale: 1.0,
            engine: EngineSignature {
                kind: "gupta-au".to_owned(),
                config_digest: gold_digest(),
                external_inputs: BTreeMap::new(),
            },
            descriptor: DescriptorSignature {
                schema: space.schema().name().to_owned(),
                version: space.schema().version(),
                hyperparameters,
                species_channels: vec![GOLD_Z],
            },
            validation_schema_version: 1,
        }
    }

    fn reference_line() -> Vec<f64> {
        let mut coordinates = vec![0.0; N * 3];
        for atom in 0..N {
            coordinates[3 * atom] = 3.0 * atom as f64;
        }
        coordinates
    }

    fn validator_config(
        space: &DescriptorSpace,
        signature: &SystemSignature,
    ) -> Result<ValidatorConfig, String> {
        let reference = reference_line();
        let described = space
            .describe(
                ArrayView1::from(reference.as_slice()),
                Some(&signature.atomic_numbers),
            )
            .map_err(|error| error.to_string())?;
        Ok(ValidatorConfig {
            reference_coordinates: reference,
            descriptor_dim: described.values().len(),
            min_separation: 1.0,
            coordinate_tolerance: 1.0e-8,
            max_gradient_norm: 1.0e-3,
            energy_abs_tolerance: 1.0e-6,
            energy_rel_tolerance: 1.0e-8,
        })
    }

    fn coordinates_within_radius(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> bool {
    if left.len() != right.len() || left.is_empty() {
        return false;
    }
    let mut peak = 0.0_f64;
    for (a, b) in left.iter().zip(right.iter()) {
        peak = peak.max((a - b).abs());
    }
    peak <= 1.0e-4
}

fn fresh_eval(gupta: &Gupta, coordinates: &[f64]) -> Result<FreshEvaluation, String> {
        if coordinates.len() != N * 3 {
            return Err("coordinate length".into());
        }
        let (energy, gradient) = gupta.value_and_gradient(ArrayView1::from(coordinates));
        let forces = gradient.iter().map(|component| -component).collect();
        Ok(FreshEvaluation { energy, forces })
    }

    fn serve(seed: u64, budget: u64) -> Result<(), String> {
        per_chain_budget(budget)?;
        let space = descriptor_space()?;
        let signature = system_signature(&space);
        let digest = signature.digest();
        let validator = validator_config(&space, &signature)?;
        let gupta = Gupta::copper_gold(0, N);
        let config = ServerConfig::new(CAMPAIGN, ensemble_name(seed), digest, CHAIN_IDS)
            .map_err(|error| error.to_string())?
            .with_scientific_state(
                signature,
                space,
                validator,
                64,
                0.35,
                budget,
                move |coordinates: &[f64]| fresh_eval(&gupta, coordinates),
            )
            .map_err(|error| error.to_string())?
            .with_exact_structure_witness(coordinates_within_radius)
            .map_err(|error| error.to_string())?;
        let server =
            CatalogServer::start("127.0.0.1:0", config).map_err(|error| error.to_string())?;
        println!("addr={}", server.addr());
        let _ = std::io::stdout().flush();
        loop {
            let _kept = server.addr();
            thread::sleep(Duration::from_secs(3600));
        }
    }

    fn check_sha(value: &str) -> Result<(), String> {
        if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            Ok(())
        } else {
            Err("libira sha256 must be 64 hex characters".into())
        }
    }

    fn check_revision(revision: &str) -> Result<(), String> {
        if revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            Ok(())
        } else {
            Err("revision must be 40 hex characters".into())
        }
    }

    fn run_seed(seed: u64, budget: u64, revision: &str, libira: &str) -> Result<(), String> {
        if !cfg!(feature = "ira") {
            return Err("seed requires the ira feature".into());
        }
        check_revision(revision)?;
        check_sha(libira)?;
        let per_chain = per_chain_budget(budget)?;
        let per_chain_usize = usize::try_from(per_chain).map_err(|_| "budget does not fit")?;
        let server = spawn_server(seed, budget)?;
        let child_pid = server.process.child.id();
        let cpu_before = self_cpu()? + proc_cpu_seconds(child_pid)?;
        let clock = Arc::new(EnsembleClock::new());
        let peak = Arc::new(AtomicU64::new(0));
        let sampler = Sampler::start(child_pid, Arc::clone(&peak));
        let gupta = Gupta::copper_gold(0, N);
        let started = Instant::now();
        let chains = run_chains(
            seed,
            server.addr,
            per_chain,
            per_chain_usize,
            &gupta,
            &clock,
        )?;
        let wall_s = started.elapsed().as_secs_f64();
        let cpu_after = self_cpu()? + proc_cpu_seconds(child_pid)?;
        drop(sampler);
        sample_peak(child_pid, &peak);
        let peak_rss = peak.load(Ordering::Acquire);
        let cpu_s = cpu_after - cpu_before;
        let status = observe(server.addr, seed)?;
        record_minima(seed, &chains)?;
        let hit_at = clock.hit_at.load(Ordering::Acquire);
        let row = build_row(&RowInput {
            seed,
            budget,
            revision,
            wall_s,
            cpu_s,
            peak_rss,
            hit_at,
            chains: &chains,
            server_aggregate: status.aggregate_charged,
            libira,
        })?;
        println!("{row}");
        Ok(())
    }

    struct ServerHandle {
        process: RunningServer,
        addr: SocketAddr,
        drain: Option<thread::JoinHandle<()>>,
    }

    impl Drop for ServerHandle {
        fn drop(&mut self) {
            let _ = self.process.child.kill();
            let _ = self.process.child.wait();
            if let Some(handle) = self.drain.take() {
                let _ = handle.join();
            }
        }
    }

    fn spawn_server(seed: u64, budget: u64) -> Result<ServerHandle, String> {
        let executable =
            std::env::current_exe().map_err(|_| "current executable is unavailable")?;
        let child = Command::new(executable)
            .args(["serve", &seed.to_string(), &budget.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|_| "catalog server failed to start")?;
        let mut process = RunningServer { child };
        let stdout = process
            .child
            .stdout
            .take()
            .ok_or("server stdout is missing")?;
        let (sender, receiver) = mpsc::channel();
        let drain = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let parsed = match reader.read_line(&mut line) {
                Ok(0) => Err("server exited before the address".to_owned()),
                Ok(_) => parse_addr(&line),
                Err(_) => Err("server address unread".to_owned()),
            };
            let _ = sender.send(parsed);
            let mut buffer = [0_u8; 8192];
            while reader.read(&mut buffer).is_ok_and(|count| count > 0) {}
        });
        let addr = match receiver.recv_timeout(Duration::from_secs(120)) {
            Ok(Ok(addr)) => addr,
            Ok(Err(error)) => {
                let _ = process.child.kill();
                let _ = drain.join();
                return Err(error);
            }
            Err(_) => {
                let _ = process.child.kill();
                let _ = drain.join();
                return Err("catalog server did not print an address".into());
            }
        };
        Ok(ServerHandle {
            process,
            addr,
            drain: Some(drain),
        })
    }

    fn parse_addr(line: &str) -> Result<SocketAddr, String> {
        line.trim()
            .strip_prefix("addr=")
            .ok_or("server address missing")?
            .parse()
            .map_err(|_| "server address invalid".to_owned())
    }

    struct Sampler {
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Sampler {
        fn start(child_pid: u32, peak: Arc<AtomicU64>) -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    sample_peak(child_pid, &peak);
                    thread::sleep(Duration::from_millis(100));
                }
                sample_peak(child_pid, &peak);
            });
            Self {
                stop,
                handle: Some(handle),
            }
        }
    }

    impl Drop for Sampler {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn sample_peak(child_pid: u32, peak: &AtomicU64) {
        let mut best = rss_anon(std::process::id()).unwrap_or(0);
        if let Some(child) = rss_anon(child_pid) {
            best = best.max(child);
        }
        peak.fetch_max(best, Ordering::Relaxed);
    }

    fn rss_anon(pid: u32) -> Option<u64> {
        let text = std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).ok()?;
        let mut anonymous = None;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("RssAnon:") {
                return kilobytes(rest);
            }
            if let Some(rest) = line.strip_prefix("Anonymous:") {
                anonymous = kilobytes(rest);
            }
        }
        anonymous
    }

    fn kilobytes(rest: &str) -> Option<u64> {
        let value: u64 = rest.split_whitespace().next()?.parse().ok()?;
        Some(value.saturating_mul(1024))
    }

    fn run_chains(
        seed: u64,
        addr: SocketAddr,
        per_chain: u64,
        per_chain_usize: usize,
        gupta: &Gupta,
        clock: &Arc<EnsembleClock>,
    ) -> Result<Vec<ChainOut>, String> {
        let space = descriptor_space()?;
        let digest = system_signature(&space).digest();
        let spaces = (0..CHAINS)
            .map(|_| descriptor_space())
            .collect::<Result<Vec<_>, _>>()?;
        thread::scope(|scope| -> Result<Vec<ChainOut>, String> {
            let mut handles = Vec::with_capacity(CHAINS);
            for (replica, space) in CHAIN_IDS.into_iter().zip(spaces) {
                let job = ChainJob {
                    replica,
                    seed,
                    addr,
                    digest,
                    per_chain,
                    per_chain_usize,
                    gupta,
                    clock: clock.as_ref(),
                    space,
                };
                handles.push(scope.spawn(move || run_chain(job)));
            }
            let mut chains = Vec::with_capacity(CHAINS);
            for handle in handles {
                match handle.join() {
                    Ok(result) => chains.push(result?),
                    Err(payload) => return Err(panic_message(payload)),
                }
            }
            Ok(chains)
        })
    }

    struct ChainJob<'a> {
        replica: u32,
        seed: u64,
        addr: SocketAddr,
        digest: [u8; 32],
        per_chain: u64,
        per_chain_usize: usize,
        gupta: &'a Gupta,
        clock: &'a EnsembleClock,
        space: DescriptorSpace,
    }

    fn panic_message(payload: Box<dyn Any + Send>) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
            })
            .unwrap_or_else(|| "chain thread panicked".to_owned())
    }

    fn run_chain(job: ChainJob<'_>) -> Result<ChainOut, String> {
        let ChainJob {
            replica,
            seed,
            addr,
            digest,
            per_chain,
            per_chain_usize,
            gupta,
            clock,
            space,
        } = job;
        let client = connect_chain(addr, replica, seed, digest)?;
        let mut cooperative =
            CooperativeRun::new([replica], per_chain).map_err(|error| error.to_string())?;
        cooperative
            .attach_client(replica, client)
            .map_err(|error| error.to_string())?;
        let cfg = search_config();
        let rng_seed = chain_seed(seed, replica);
        let start = start_state(&cfg, rng_seed)?;
        let mut ledger = Ledger::with_stop(per_chain_usize, Arc::clone(&clock.stop));
        ledger.stop_below(REFERENCE + DEFAULT_HIT_TOL);
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let mut optimiser = WarmLbfgs::default();
        let replica_index = usize::try_from(replica).expect("replica");
        let mut slice = SliceState::new(replica, rng_seed, &mut cooperative, space);
        let outcome = {
            let mut last_point = Array1::<f64>::zeros(0);
            let mut last_gradient = Array1::<f64>::zeros(0);
            let mut relax = |led: &mut Ledger, point: ArrayView1<f64>, iterations: usize| {
                let charged_before = led.spent();
                optimiser.forget();
                let (value, relaxed, _) = optimiser.minimize(point, iterations, |probe| {
                    if !led.charge() {
                        return None;
                    }
                    let (energy, gradient) = gupta.value_and_gradient(probe);
                    if last_point.len() != probe.len() {
                        last_point = Array1::zeros(probe.len());
                        last_gradient = Array1::zeros(probe.len());
                    }
                    last_point.assign(&probe);
                    last_gradient.assign(&gradient);
                    Some((energy, gradient))
                });
                clock.note_quench(replica_index, led.spent(), value);
                let evidence = quench_gradient(&last_point, &last_gradient, relaxed.view());
                let _ = led.record_quench_boundary(
                    charged_before,
                    value,
                    relaxed.clone(),
                    evidence,
                );
                (value, relaxed)
            };
            let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
                if !led.charge() {
                    return None;
                }
                let gradient = gupta.value_and_gradient(point).1;
                clock.publish(replica_index, led.spent());
                Some(gradient)
            };
            let mut checkpoint = |snapshot: ChainCheckpoint<'_>| match slice.on_checkpoint(snapshot)
            {
                Ok(action) => action,
                Err(error) => panic!("{error}"),
            };
            run_with_energy_gradient_at_checkpoints(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                None,
                &mut rng,
                SLICE,
                &mut checkpoint,
            )
            .map_err(|error| error.to_string())?
        };
        let charged = outcome.charged;
        let best = outcome.best;
        let best_state = outcome.best_state;
        slice.settle_tail(ledger.quench_boundaries(), charged)?;
        slice.flush()?;
        let (summaries, ensemble_total) = summarize(slice.cooperative.ledger());
        Ok(ChainOut {
            replica,
            best,
            best_state,
            charged,
            summaries,
            ensemble_total,
        })
    }

    fn connect_chain(
        addr: SocketAddr,
        replica: u32,
        seed: u64,
        digest: [u8; 32],
    ) -> Result<CatalogClient, String> {
        let identity = CatalogIdentity {
            campaign: CAMPAIGN.to_owned(),
            ensemble: ensemble_name(seed),
            replica,
            signature_digest: digest,
        };
        let mut last = String::from("catalog connect failed");
        for _ in 0..50 {
            match CatalogClient::connect(addr, identity.clone(), ClientConfig::default()) {
                Ok(mut client) => {
                    client.attach(1).map_err(|error| error.to_string())?;
                    return Ok(client);
                }
                Err(error) => {
                    last = error.to_string();
                    thread::sleep(Duration::from_millis(20));
                }
            }
        }
        Err(last)
    }

    fn search_config() -> Config {
        let mut cfg = Config::communicating(N).with_species(Gupta::copper_gold_species(0, N), None);
        cfg.replicas = 1;
        cfg.basin_stride = 8;
        cfg.polish_records = cfg.relax_steps;
        cfg.shape_keyed = true;
        cfg
    }

    fn start_state(cfg: &Config, rng_seed: u64) -> Result<Array1<f64>, String> {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let start = random_cluster_in_radius(N, cfg.start_radius(), cfg.min_separation, &mut rng);
        if start.len() != N * 3 {
            return Err("cluster placement did not finish".into());
        }
        Ok(start)
    }

    struct SliceState<'a> {
        replica: u32,
        rng_seed: u64,
        cooperative: &'a mut CooperativeRun,
        space: DescriptorSpace,
        species: Vec<u32>,
        last_charged: usize,
        recorded_boundaries: usize,
        slice_index: u64,
        stall: u32,
        last_best: f64,
        sequence: u64,
    }

    impl<'a> SliceState<'a> {
        fn new(
            replica: u32,
            rng_seed: u64,
            cooperative: &'a mut CooperativeRun,
            space: DescriptorSpace,
        ) -> Self {
            Self {
                replica,
                rng_seed,
                cooperative,
                space,
                species: vec![GOLD_Z; N],
                last_charged: 0,
                recorded_boundaries: 0,
                slice_index: 0,
                stall: 0,
                last_best: f64::INFINITY,
                sequence: 0,
            }
        }

        fn on_checkpoint(
            &mut self,
            snapshot: ChainCheckpoint<'_>,
        ) -> Result<CheckpointAction, String> {
            let charged = snapshot.charged();
            if charged < self.last_charged {
                return Err("charged counter went backwards".into());
            }
            let delta = charged - self.last_charged;
            let boundaries = snapshot.quench_boundaries();
            let boundary_calls = self.push_boundaries(boundaries)?;
            self.recorded_boundaries = self.recorded_boundaries.saturating_add(boundaries.len());
            if boundary_calls > delta {
                return Err(format!(
                    "quench calls {boundary_calls} exceed slice {delta}"
                ));
            }
            self.charge_auxiliary(delta - boundary_calls)?;
            self.last_charged = charged;
            for boundary in boundaries {
                self.offer_boundary(boundary, charged)?;
            }
            let (role, reason) = self.policy_role(&snapshot)?;
            let proposal = self.deeper_sample(&snapshot)?;
            if proposal.is_some() {
                self.charge(ChargeKind::RemoteProposal, 0)?;
            }
            let family = if proposal.is_some() {
                ProposalFamily::CatalogSample
            } else {
                ProposalFamily::Local
            };
            let step = proposal
                .as_ref()
                .and_then(|state| step_norm(snapshot.current_state(), state));
            self.push_slice(&snapshot, delta, role, reason, family, step)?;
            let action = if let Some(state) = proposal {
                CheckpointAction::BoundaryProposal {
                    state,
                    action: "catalog-sample".to_owned(),
                }
            } else {
                CheckpointAction::Continue
            };
            Ok(action)
        }

        fn settle_tail(
            &mut self,
            boundaries: &[QuenchBoundary],
            spent: usize,
        ) -> Result<(), String> {
            if self.recorded_boundaries > boundaries.len() {
                return Err("checkpoint boundaries exceed the hop ledger".into());
            }
            let tail = &boundaries[self.recorded_boundaries..];
            let boundary_calls = self.push_boundaries(tail)?;
            if spent < self.last_charged {
                return Err("hop ledger spent went backwards".into());
            }
            let delta = spent - self.last_charged;
            if boundary_calls > delta {
                return Err(format!(
                    "tail quench calls {boundary_calls} exceed remaining {delta}"
                ));
            }
            self.charge_auxiliary(delta - boundary_calls)?;
            self.last_charged = spent;
            for boundary in tail {
                self.offer_boundary(boundary, spent)?;
            }
            Ok(())
        }

        fn flush(&mut self) -> Result<(), String> {
            self.cooperative
                .flush(self.replica)
                .map_err(|error| error.to_string())
        }

        fn push_boundaries(&mut self, boundaries: &[QuenchBoundary]) -> Result<usize, String> {
            let mut calls = 0usize;
            let mut work = Vec::new();
            for boundary in boundaries {
                let count = boundary.charged_calls();
                if count == 0 {
                    continue;
                }
                calls = calls.saturating_add(count);
                let kind = match boundary.status() {
                    QuenchStatus::Validated => ChargeKind::AcceptedQuench,
                    QuenchStatus::Rejected => ChargeKind::RejectedQuench,
                };
                work.push((
                    kind,
                    u64::try_from(count).map_err(|_| "call count".to_owned())?,
                ));
            }
            if !work.is_empty() {
                self.cooperative
                    .record_work_batch(self.replica, work)
                    .map_err(|error| error.to_string())?;
            }
            Ok(calls)
        }

        fn charge_auxiliary(&mut self, calls: usize) -> Result<(), String> {
            if calls == 0 {
                return Ok(());
            }
            self.charge(ChargeKind::AuxiliaryEvaluation, calls)
        }

        fn charge(&mut self, kind: ChargeKind, calls: usize) -> Result<(), String> {
            let calls = u64::try_from(calls).map_err(|_| "call count".to_owned())?;
            self.cooperative
                .record_work(self.replica, kind, calls)
                .map_err(|error| error.to_string())
        }

        fn offer_boundary(
            &mut self,
            boundary: &QuenchBoundary,
            charged: usize,
        ) -> Result<(), String> {
            if boundary.status() != QuenchStatus::Validated {
                return Ok(());
            }
            let Some(gradient) = boundary.gradient() else {
                return Ok(());
            };
            let Some(candidate) =
                self.candidate(boundary.energy(), boundary.state(), gradient, charged)?
            else {
                return Ok(());
            };
            self.charge(ChargeKind::DescriptorEvaluation, 0)?;
            match self
                .cooperative
                .offer_candidate(self.replica, candidate)
                .map_err(|error| error.to_string())?
            {
                CatalogOfferOutcome::LocalFallback | CatalogOfferOutcome::SharingDisabled => {
                    self.charge(ChargeKind::RpcFallback, 0)?;
                }
                CatalogOfferOutcome::Admitted | CatalogOfferOutcome::Rejected => {}
            }
            Ok(())
        }

        fn candidate(
            &mut self,
            energy: f64,
            coordinates: ArrayView1<f64>,
            gradient: ArrayView1<f64>,
            charged: usize,
        ) -> Result<Option<CatalogCandidate>, String> {
            if !energy.is_finite() || coordinates.len() != gradient.len() {
                return Ok(None);
            }
            let Some(components) = gradient.as_slice() else {
                return Ok(None);
            };
            let norm = euclidean_gradient_norm(components);
            if !norm.is_finite() || norm > 1.0e-3 {
                return Ok(None);
            }
            let described = self
                .space
                .describe(coordinates, Some(&self.species))
                .map_err(|error| error.to_string())?;
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or("event sequence overflow")?;
            Ok(Some(CatalogCandidate {
                producer_replica: self.replica,
                coordinates: coordinates.to_vec(),
                cell: None,
                energy,
                forces: gradient.iter().map(|component| -component).collect(),
                gradient_norm: norm,
                descriptor: described.values().to_vec(),
                descriptor_schema_version: described.schema_version(),
                quench_converged: true,
                charged_work: u64::try_from(charged).map_err(|_| "charged work".to_owned())?,
                event_sequence: self.sequence,
                seed: self.rng_seed,
                census_basin: None,
            }))
        }

        fn policy_role(
            &mut self,
            snapshot: &ChainCheckpoint<'_>,
        ) -> Result<(PolicyRole, &'static str), String> {
            let energy = snapshot.current_energy();
            if !energy.is_finite() {
                return Ok((PolicyRole::Unavailable, "nonfinite"));
            }
            let described = self
                .space
                .describe(snapshot.current_state(), Some(&self.species))
                .map_err(|error| error.to_string())?;
            self.charge(ChargeKind::DescriptorEvaluation, 0)?;
            let best = snapshot.best_energy();
            let deepened = best.is_finite() && best < self.last_best - 1.0e-12;
            if deepened {
                self.stall = 0;
                self.last_best = best;
            } else {
                self.stall = self.stall.saturating_add(1);
            }
            let outcome = self
                .cooperative
                .policy_input(
                    self.replica,
                    described.values().to_vec(),
                    energy,
                    self.stall,
                    deepened,
                )
                .map_err(|error| error.to_string())?;
            match outcome {
                PolicyEvidenceOutcome::Remote(input) => {
                    let decision = self
                        .cooperative
                        .decide(self.replica, input)
                        .map_err(|error| error.to_string())?;
                    Ok((role_of(decision.action), decision.reason.code()))
                }
                PolicyEvidenceOutcome::LocalFallback => {
                    self.charge(ChargeKind::RpcFallback, 0)?;
                    Ok((PolicyRole::Unavailable, "rpc_fallback"))
                }
                PolicyEvidenceOutcome::SharingDisabled => {
                    self.charge(ChargeKind::RpcFallback, 0)?;
                    Ok((PolicyRole::Unavailable, "sharing_disabled"))
                }
                PolicyEvidenceOutcome::Rejected => Ok((PolicyRole::Unavailable, "rejected")),
            }
        }

        fn deeper_sample(
            &mut self,
            snapshot: &ChainCheckpoint<'_>,
        ) -> Result<Option<Array1<f64>>, String> {
            let first = self
                .cooperative
                .try_sample_candidate(self.replica, INCUMBENT_SAMPLE_DRAW)
                .map_err(|error| error.to_string())?;
            let outcome = if matches!(first, CatalogSampleOutcome::Candidate(_)) {
                first
            } else {
                self.cooperative
                    .try_sample_candidate(self.replica, INCUMBENT_SAMPLE_DRAW)
                    .map_err(|error| error.to_string())?
            };
            let CatalogSampleOutcome::Candidate(held) = outcome else {
                return Ok(None);
            };
            if held.coordinates.len() != snapshot.current_state().len() {
                return Ok(None);
            }
            let best = snapshot.best_energy();
            if best.is_finite() && held.energy < best - DEFAULT_HIT_TOL {
                Ok(Some(Array1::from(held.coordinates)))
            } else {
                Ok(None)
            }
        }

        fn push_slice(
            &mut self,
            snapshot: &ChainCheckpoint<'_>,
            delta: usize,
            role: PolicyRole,
            reason: &'static str,
            family: ProposalFamily,
            step: Option<f64>,
        ) -> Result<(), String> {
            self.slice_index = self
                .slice_index
                .checked_add(1)
                .ok_or("slice counter overflow")?;
            let energy = snapshot.best_energy();
            let diagnostic = SliceTrace {
                slice: self.slice_index,
                current_basin: None,
                active_relation: None,
                policy_role: role,
                policy_reason: reason,
                proposal_family: family,
                sampled_basin: None,
                descriptor_step_norm: None,
                cartesian_step_norm: step,
                validation: SliceValidation::NotAttempted,
                quench: SliceQuench::NotAttempted,
                adoption: SliceAdoption::NotAttempted,
                novelty: None,
                energy: energy.is_finite().then_some(energy),
                charged_work: u64::try_from(delta).map_err(|_| "slice work".to_owned())?,
            };
            self.cooperative
                .record_slice(self.replica, diagnostic)
                .map_err(|error| error.to_string())
        }
    }

    fn role_of(action: PolicyAction) -> PolicyRole {
        match action {
            PolicyAction::ContinueLocal => PolicyRole::Local,
            PolicyAction::Exploit { .. } => PolicyRole::Exploit,
            PolicyAction::Explore => PolicyRole::Explore,
            PolicyAction::Leave => PolicyRole::Leave,
        }
    }

    fn quench_gradient(
    last_point: &Array1<f64>,
    last_gradient: &Array1<f64>,
    relaxed: ArrayView1<f64>,
) -> Option<Array1<f64>> {
    if last_point.len() != relaxed.len() || last_point.is_empty() {
        return None;
    }
    if last_point
        .iter()
        .zip(relaxed.iter())
        .any(|(seen, value)| seen.to_bits() != value.to_bits())
    {
        return None;
    }
    let norm = last_gradient
        .as_slice()
        .map(euclidean_gradient_norm)
        .unwrap_or(f64::INFINITY);
    (norm.is_finite() && norm <= 1.0e-3).then(|| last_gradient.clone())
}

fn step_norm(current: ArrayView1<f64>, proposed: &Array1<f64>) -> Option<f64> {
        if current.len() != proposed.len() || current.is_empty() {
            return None;
        }
        let mut sum = 0.0;
        for (left, right) in current.iter().zip(proposed.iter()) {
            let delta = left - right;
            sum += delta * delta;
        }
        let norm = (sum / current.len() as f64).sqrt();
        norm.is_finite().then_some(norm)
    }

    fn summarize(ledger: &CooperativeLedger) -> ([ChargeSummary; 11], u64) {
        let mut summaries = [ChargeSummary::default(); 11];
        for (slot, kind) in summaries.iter_mut().zip(KINDS) {
            *slot = ledger.charge_summary(kind);
        }
        (summaries, ledger.ensemble_total())
    }

    fn observe(
        addr: SocketAddr,
        seed: u64,
    ) -> Result<anneal_core::catalog_rpc::CoordinatorStatus, String> {
        let space = descriptor_space()?;
        let digest = system_signature(&space).digest();
        let identity = CatalogIdentity {
            campaign: CAMPAIGN.to_owned(),
            ensemble: ensemble_name(seed),
            replica: u32::MAX,
            signature_digest: digest,
        };
        let mut client = CatalogClient::connect(addr, identity, ClientConfig::default())
            .map_err(|error| error.to_string())?;
        client.observer_status(1).map_err(|error| error.to_string())
    }

    fn record_minima(seed: u64, chains: &[ChainOut]) -> Result<(), String> {
        let space = descriptor_space()?;
        let digest = system_signature(&space).digest();
        let root = minima_root()?;
        let corpus = MinimaCorpus::open(&root, digest).map_err(|error| error.to_string())?;
        let units = MinimaUnits {
            length: "angstrom".to_owned(),
            energy: "eV".to_owned(),
        };
        let set = MinimaSet {
            system: "au55".to_owned(),
            temperature: search_config().temperature,
            seed,
        };
        let species = Gupta::copper_gold_species(0, N);
        for chain in chains {
            let state = chain
                .best_state
                .as_ref()
                .ok_or("chain finished without a state")?;
            if !chain.best.is_finite() {
                return Err("chain best energy is nonfinite".into());
            }
            let provenance = serde_json::json!({
                "arm": "cooperative",
                "replica": chain.replica,
                "seed": seed,
            });
            corpus
                .record(
                    &set,
                    &species,
                    &units,
                    &[(chain.best, state.view())],
                    0.0,
                    provenance,
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn minima_root() -> Result<PathBuf, String> {
        let root = match std::env::var("MINIMA_ROOT") {
            Ok(value) if !value.is_empty() => PathBuf::from(value),
            _ => std::env::temp_dir().join("au55-cooperative-minima"),
        };
        std::fs::create_dir_all(&root).map_err(|_| "minima directory could not be created")?;
        Ok(root)
    }

    #[derive(Clone, Copy)]
    struct RowInput<'a> {
        seed: u64,
        budget: u64,
        revision: &'a str,
        wall_s: f64,
        cpu_s: f64,
        peak_rss: u64,
        hit_at: u64,
        chains: &'a [ChainOut],
        server_aggregate: u64,
        libira: &'a str,
    }

    fn build_row(input: &RowInput<'_>) -> Result<String, String> {
        let RowInput {
            seed,
            budget,
            revision,
            wall_s,
            cpu_s,
            peak_rss,
            hit_at,
            chains,
            server_aggregate,
            libira,
        } = *input;
        let mut summaries = [ChargeSummary::default(); 11];
        let mut oracle_calls = 0u64;
        let mut ensemble_total = 0u64;
        let mut best = f64::INFINITY;
        for chain in chains {
            oracle_calls = oracle_calls
                .checked_add(u64::try_from(chain.charged).map_err(|_| "oracle calls")?)
                .ok_or("oracle calls overflow")?;
            ensemble_total = ensemble_total
                .checked_add(chain.ensemble_total)
                .ok_or("ensemble total overflow")?;
            if chain.best < best {
                best = chain.best;
            }
            for (slot, part) in summaries.iter_mut().zip(chain.summaries) {
                slot.events = slot
                    .events
                    .checked_add(part.events)
                    .ok_or("event overflow")?;
                slot.charged_calls = slot
                    .charged_calls
                    .checked_add(part.charged_calls)
                    .ok_or("charge overflow")?;
            }
        }
        let charge_sum = summaries
            .iter()
            .try_fold(0u64, |sum, summary| sum.checked_add(summary.charged_calls));
        let charge_sum = charge_sum.ok_or("charge sum overflow")?;
        let best_hits = chains
            .iter()
            .any(|chain| energy_hits(chain.best, REFERENCE, DEFAULT_HIT_TOL));
        let (hit, calls_to_hit, hit_clock) = hit_fields(hit_at, best_hits, oracle_calls, budget);
        let terms_match = u8::from(
            charge_sum == ensemble_total
                && ensemble_total == oracle_calls
                && oracle_calls == server_aggregate,
        );
        let mut fields = vec![
            "au55".to_owned(),
            "cooperative".to_owned(),
            seed.to_string(),
            CHAINS.to_string(),
            oracle_calls.to_string(),
            budget.to_string(),
            hit.to_string(),
            calls_to_hit.to_string(),
            hit_clock.to_owned(),
            format!("{best:.10}"),
            format!("{wall_s:.6}"),
            format!("{cpu_s:.6}"),
            peak_rss.to_string(),
        ];
        for summary in summaries {
            fields.push(summary.events.to_string());
            fields.push(summary.charged_calls.to_string());
        }
        fields.push(charge_sum.to_string());
        fields.push(ensemble_total.to_string());
        fields.push(server_aggregate.to_string());
        fields.push(terms_match.to_string());
        fields.push("arch_cooperative".to_owned());
        fields.push("bank-rpc ira".to_owned());
        fields.push("true".to_owned());
        fields.push(libira.to_owned());
        fields.push(revision.to_owned());
        fields.push(nproc().to_string());
        fields.push(cpu_model());
        Ok(fields.join(","))
    }

    fn hit_fields(
        hit_at: u64,
        best_hits: bool,
        oracle_calls: u64,
        budget: u64,
    ) -> (u8, u64, &'static str) {
        if hit_at != u64::MAX {
            (1, hit_at, "quench")
        } else if best_hits {
            eprintln!(
                "hit energy without a quench snapshot; calls_to_hit uses the end charged sum"
            );
            (1, oracle_calls, "end")
        } else {
            (0, budget.saturating_add(1), "miss")
        }
    }

    fn nproc() -> usize {
        let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let processors = text
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        if processors > 0 {
            processors
        } else {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        }
    }

    fn cpu_model() -> String {
        let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let model = text
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == "model name").then(|| value.trim().to_owned())
            })
            .unwrap_or_else(|| "unknown".to_owned());
        model
            .chars()
            .map(|glyph| {
                if glyph == ',' || glyph == '"' || glyph.is_control() {
                    ' '
                } else {
                    glyph
                }
            })
            .collect()
    }

    fn self_cpu() -> Result<f64, String> {
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        // SAFETY: getrusage writes a single rusage for this process and the
        // pointer is a live local.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return Err("getrusage failed".into());
        }
        Ok(timeval_secs(usage.ru_utime) + timeval_secs(usage.ru_stime))
    }

    fn timeval_secs(value: libc::timeval) -> f64 {
        value.tv_sec as f64 + value.tv_usec as f64 * 1.0e-6
    }

    fn proc_cpu_seconds(pid: u32) -> Result<f64, String> {
        let text = std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .map_err(|_| "process stat unreadable")?;
        let rest = text
            .rsplit_once(')')
            .ok_or("process stat has no command end")?
            .1;
        let fields: Vec<&str> = rest.split_whitespace().collect();
        if fields.len() <= 12 {
            return Err("process stat is short".into());
        }
        let utime: u64 = fields[11].parse().map_err(|_| "utime")?;
        let stime: u64 = fields[12].parse().map_err(|_| "stime")?;
        Ok((utime + stime) as f64 / ticks_per_second()?)
    }

    fn ticks_per_second() -> Result<f64, String> {
        // SAFETY: _SC_CLK_TCK takes no pointer and returns a process constant.
        let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if ticks <= 0 {
            return Err("clock tick query failed".into());
        }
        Ok(ticks as f64)
    }
}
