//! Conformer search of a flexible molecule at GFN2-xTB.
//!
//! Several chains divide one charged budget. Each reports every certified
//! quench to one exact minimum history, so a chain that lands where another
//! has been pays that chain's visits into its own bias. The chains take
//! turns on one core: a chain hands the turn on at every potential call.
//! The run is therefore as sequential as a single chain, and the history
//! sees the same order on every rerun. The validated minima of every chain
//! go to a readcon-db corpus. The xyz file the scorer reads is read back
//! from that corpus.
//!
//! Nothing here knows a reference conformer. Every chain starts from the
//! file on the command line.
//!
//! Usage: conformer_search <xyz> <budget> <seed> <prefix> [options]
//!
//!   --replicas R         chains dividing the budget (default 1)
//!   --history MODE       shared, private or none (default shared)
//!   --identity A         heavy-atom RMSD in angstrom up to which two
//!                        minima are one (default 0.5)
//!   --polish N           descent steps that settle each new best (default:
//!                        the preset's)
//!   --precondition B     on or off: bond-graph preconditioned quenches
//!                        (default on)
//!   --height-revisits K  revisits a basin takes to fill with bias
//!                        (default: the preset's)
//!   --settle F           share of the budget held back from the search to
//!                        settle the best minimum to 1e-3 eV/angstrom once
//!                        every chain has finished (default 0.1)
//!   --corpus DIR         readcon-db root (default <prefix>.corpus)
//!
//! The engine is the persistent helper in `ase_objective.py`
//! (`ASE_ENGINE`, default `xtb-cli`; `XTB_EXE` names the binary).

#[path = "common/pipe_engine.rs"]
mod pipe_engine;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use anneal_core::catalog::{DescriptorSignature, EngineSignature, SystemSignature};
use anneal_core::descriptor_space::{
    DescriptorGeometry, DescriptorSpace, universal_descriptor_space,
};
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, MoveLibrary, Outcome, QuenchStatus, connectivity_groups_z,
};
use anneal_core::methods::cluster_search::{RelaxStats, Settled, search_from_with_history, settle};
use anneal_core::methods::minima_hopping::{
    HistoryHook, HistoryMembership, MinimumHistory, SharedMinimumHistory,
};
use anneal_core::minima_db::{MinimaCorpus, MinimaSet, MinimaUnits};
use anneal_core::pes_exploration::StructureContext;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};
use pipe_engine::PipeEngine;
use readcon_core::helpers::symbol_to_atomic_number;
use sha2::{Digest, Sha256};

const HARTREE_EV: f64 = 27.211386245988;
/// Gradient component, in eV/angstrom, a settled minimum ends below: a
/// fiftieth of the molecular record gate.
const SETTLE_GRADIENT: f64 = 1e-3;
const HELPER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/ase_objective.py");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum History {
    Shared,
    Private,
    None,
}

impl History {
    fn parse(value: &str) -> Self {
        match value {
            "shared" => Self::Shared,
            "private" => Self::Private,
            "none" => Self::None,
            other => panic!("--history must be shared, private or none; got {other}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Private => "private",
            Self::None => "none",
        }
    }
}

struct Options {
    xyz: String,
    budget: usize,
    seed: u64,
    prefix: String,
    replicas: usize,
    history: History,
    identity: f64,
    polish: Option<usize>,
    precondition: Option<bool>,
    height_revisits: Option<f64>,
    settle: f64,
    corpus: PathBuf,
}

fn options() -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: conformer_search <xyz> <budget> <seed> <prefix> [--replicas R] \
                 [--history shared|private|none] [--identity A] [--polish N] \
                 [--precondition on|off] [--height-revisits K] [--settle F] [--corpus DIR]";
    let positional: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    assert_eq!(positional.len(), 4, "{usage}");
    let prefix = positional[3].clone();
    let mut out = Options {
        xyz: positional[0].clone(),
        budget: positional[1].parse().expect("budget"),
        seed: positional[2].parse().expect("seed"),
        corpus: PathBuf::from(format!("{prefix}.corpus")),
        prefix,
        replicas: 1,
        history: History::Shared,
        identity: 0.5,
        polish: None,
        precondition: Some(true),
        height_revisits: None,
        settle: 0.1,
    };
    let mut rest = args[4..].iter();
    while let Some(flag) = rest.next() {
        let value = rest
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value; {usage}"));
        match flag.as_str() {
            "--replicas" => out.replicas = value.parse().expect("replicas"),
            "--history" => out.history = History::parse(value),
            "--identity" => out.identity = value.parse().expect("identity"),
            "--polish" => out.polish = Some(value.parse().expect("polish steps")),
            "--precondition" => {
                out.precondition = Some(match value.as_str() {
                    "on" => true,
                    "off" => false,
                    other => panic!("--precondition must be on or off; got {other}"),
                })
            }
            "--height-revisits" => {
                out.height_revisits = Some(value.parse().expect("height revisits"))
            }
            "--settle" => out.settle = value.parse().expect("settle share"),
            "--corpus" => out.corpus = PathBuf::from(value),
            other => panic!("unknown option {other}; {usage}"),
        }
    }
    assert!(out.replicas > 0, "at least one replica");
    assert!(
        (0.0..1.0).contains(&out.settle),
        "the settle share must lie in [0, 1)"
    );
    assert!(
        out.budget >= out.replicas + out.reserve(),
        "every replica needs a call beyond the settle reserve"
    );
    assert!(out.identity > 0.0, "the identity radius must be positive");
    out
}

impl Options {
    /// Calls held back from the search to settle the best.
    fn reserve(&self) -> usize {
        (self.settle * self.budget as f64).floor() as usize
    }
}

/// Whose turn it is to run, among the chains still running.
///
/// Exactly one chain runs at a time. A chain hands the turn to the next
/// live one before every potential call and waits for it to come back, so
/// the calls interleave in a fixed order.
struct Turns {
    state: Mutex<(usize, Vec<bool>)>,
    ready: Condvar,
}

impl Turns {
    fn new(chains: usize) -> Self {
        Self {
            state: Mutex::new((0, vec![true; chains])),
            ready: Condvar::new(),
        }
    }

    fn next_live(live: &[bool], from: usize) -> usize {
        (1..=live.len())
            .map(|k| (from + k) % live.len())
            .find(|&c| live[c])
            .unwrap_or(from)
    }

    fn wait(&self, chain: usize) {
        let mut state = self.state.lock().expect("turns");
        while state.0 != chain {
            state = self.ready.wait(state).expect("turns");
        }
    }

    fn hand_on(&self, chain: usize) {
        {
            let mut state = self.state.lock().expect("turns");
            state.0 = Self::next_live(&state.1, chain);
        }
        self.ready.notify_all();
        self.wait(chain);
    }

    fn retire(&self, chain: usize) {
        {
            let mut state = self.state.lock().expect("turns");
            state.1[chain] = false;
            if state.0 == chain {
                state.0 = Self::next_live(&state.1, chain);
            }
        }
        self.ready.notify_all();
    }
}

/// Retires a chain however its thread ends, so a panic cannot stall the rest.
struct Retire<'a>(&'a Turns, usize);

impl Drop for Retire<'_> {
    fn drop(&mut self) {
        self.0.retire(self.1);
    }
}

/// One line per potential call across all chains: call, seconds, energy.
struct CallLog {
    started: Instant,
    calls: AtomicUsize,
    trace: Mutex<BufWriter<File>>,
}

impl CallLog {
    fn record(&self, energy: f64) {
        let mut trace = self.trace.lock().expect("trace");
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let t = self.started.elapsed().as_secs_f64();
        let _ = writeln!(trace, "{call} {t:.6} {energy:.10}");
    }
}

/// GFN2-xTB through the helper. Every method is one engine call.
struct Xtb<'a> {
    engine: Mutex<PipeEngine>,
    bounds: Bounds<f64>,
    chain: usize,
    turns: &'a Turns,
    log: &'a CallLog,
}

impl Xtb<'_> {
    fn call(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.turns.hand_on(self.chain);
        let got = self.engine.lock().expect("engine").eval(x);
        let (energy, gradient) = got.unwrap_or_else(|| (f64::INFINITY, Array1::zeros(x.len())));
        self.log.record(energy);
        (energy, gradient)
    }
}

impl Objective<f64> for Xtb<'_> {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.call(x).0
    }
}

impl Gradient<f64> for Xtb<'_> {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.call(x).1
    }

    fn dim(&self) -> usize {
        self.bounds.dims
    }
}

impl DifferentiableObjective<f64> for Xtb<'_> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.call(x)
    }
}

/// Heavy-atom root-mean-square deviation after the best proper rotation.
///
/// Horn's closed form (J. Opt. Soc. Am. A 4, 629, 1987): the largest
/// eigenvalue of a 4 by 4 quaternion matrix. A unit quaternion is a proper
/// rotation, so a mirror image of a chiral conformer stays apart. Atom
/// order is fixed; hydrogens are left out, so a turned methyl is the same
/// conformer.
struct HeavyAtomRmsd {
    heavy: Vec<usize>,
}

impl HeavyAtomRmsd {
    fn new(species: &[u32]) -> Self {
        let heavy: Vec<usize> = (0..species.len()).filter(|&i| species[i] != 1).collect();
        assert!(!heavy.is_empty(), "no heavy atom to align");
        Self { heavy }
    }

    fn centred(&self, x: ArrayView1<f64>) -> Vec<[f64; 3]> {
        let n = self.heavy.len() as f64;
        let mut c = [0.0; 3];
        for &i in &self.heavy {
            for k in 0..3 {
                c[k] += x[3 * i + k] / n;
            }
        }
        self.heavy
            .iter()
            .map(|&i| [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]])
            .collect()
    }

    fn rmsd(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let (a, b) = (self.centred(a), self.centred(b));
        let mut s = [[0.0; 3]; 3];
        let mut inner = 0.0;
        for (p, q) in a.iter().zip(&b) {
            for (i, row) in s.iter_mut().enumerate() {
                inner += p[i] * p[i] + q[i] * q[i];
                for (cell, qj) in row.iter_mut().zip(q) {
                    *cell += p[i] * qj;
                }
            }
        }
        let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
        let n = [
            [xx + yy + zz, yz - zy, zx - xz, xy - yx],
            [yz - zy, xx - yy - zz, xy + yx, zx + xz],
            [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
            [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
        ];
        let msd = (inner - 2.0 * largest_eigenvalue(n)) / a.len() as f64;
        msd.max(0.0).sqrt()
    }
}

/// Largest eigenvalue of a symmetric 4 by 4 matrix by cyclic Jacobi sweeps.
fn largest_eigenvalue(mut m: [[f64; 4]; 4]) -> f64 {
    for _ in 0..64 {
        let off: f64 = (0..4)
            .flat_map(|p| (p + 1..4).map(move |q| (p, q)))
            .map(|(p, q)| m[p][q] * m[p][q])
            .sum();
        let diagonal: f64 = (0..4).map(|p| m[p][p] * m[p][p]).sum();
        if off <= 1e-30 * diagonal.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if m[p][q] == 0.0 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in &mut m {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
                let [row_p, row_q] = m.get_disjoint_mut([p, q]).expect("p < q");
                for (pk, qk) in row_p.iter_mut().zip(row_q.iter_mut()) {
                    (*pk, *qk) = (c * *pk - s * *qk, s * *pk + c * *qk);
                }
            }
        }
    }
    (0..4).map(|p| m[p][p]).fold(f64::NEG_INFINITY, f64::max)
}

fn read_xyz(path: &str) -> (Vec<String>, Array1<f64>, Vec<u32>) {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut lines = BufReader::new(file).lines().map(|l| l.expect("xyz line"));
    let n: usize = lines
        .next()
        .expect("xyz header")
        .trim()
        .parse()
        .expect("atom count");
    let _ = lines.next();
    let mut symbols = Vec::with_capacity(n);
    let mut x = Array1::zeros(3 * n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let line = lines.next().unwrap_or_else(|| panic!("atom {i} missing"));
        let mut parts = line.split_whitespace();
        let symbol = parts.next().expect("symbol").to_string();
        for k in 0..3 {
            x[3 * i + k] = parts
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("atom {i} coordinate {k}"));
        }
        let number = symbol_to_atomic_number(&symbol);
        assert!(number > 0, "no atomic number for {symbol}");
        z.push(u32::try_from(number).expect("atomic number"));
        symbols.push(symbol);
    }
    (symbols, x, z)
}

fn write_frame(out: &mut impl Write, symbols: &[String], x: ArrayView1<f64>, comment: &str) {
    let _ = writeln!(out, "{}\n{comment}", symbols.len());
    for (i, symbol) in symbols.iter().enumerate() {
        let _ = writeln!(
            out,
            "{symbol} {:.8} {:.8} {:.8}",
            x[3 * i],
            x[3 * i + 1],
            x[3 * i + 2]
        );
    }
}

fn sha256_file(path: &std::path::Path) -> [u8; 32] {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    Sha256::digest(&bytes).into()
}

fn xtb_binary() -> PathBuf {
    if let Ok(exe) = std::env::var("XTB_EXE") {
        return PathBuf::from(exe);
    }
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join("xtb"))
                .find(|candidate| candidate.is_file())
        })
        .expect("set XTB_EXE or put xtb on PATH")
}

/// The system a corpus belongs to: atoms, fragments, engine and descriptor.
fn signature(
    species: &[u32],
    fragments: &[Vec<usize>],
    engine: &str,
    descriptor: &DescriptorSpace,
    length_scale: f64,
) -> SystemSignature {
    let n = species.len();
    let mut labels = vec![0u32; n];
    for (label, fragment) in fragments.iter().enumerate() {
        for &atom in fragment {
            labels[atom] = u32::try_from(label).expect("fragment label");
        }
    }
    let mut external_inputs = BTreeMap::new();
    external_inputs.insert("xtb".to_string(), sha256_file(&xtb_binary()));
    let mut hyperparameters = BTreeMap::new();
    hyperparameters.insert(
        "geometry".to_string(),
        format!("finite;length-scale={length_scale}"),
    );
    let mut channels = species.to_vec();
    channels.sort_unstable();
    channels.dedup();
    SystemSignature {
        atomic_numbers: species.to_vec(),
        coordinate_dim: 3 * n as u64,
        group_labels: labels,
        group_schema: "covalent-fragments-v1".into(),
        frozen_mask: vec![false; n],
        cell: None,
        periodic: [false; 3],
        length_scale: 1.0,
        energy_scale: 1.0,
        engine: EngineSignature {
            kind: format!("ase-helper:{engine}"),
            config_digest: sha256_file(std::path::Path::new(HELPER)),
            external_inputs,
        },
        descriptor: DescriptorSignature {
            schema: descriptor.schema().name().into(),
            version: descriptor.schema().version(),
            hyperparameters,
            species_channels: channels,
        },
        validation_schema_version: 1,
    }
}

struct ChainReport {
    seed: u64,
    budget: usize,
    outcome: Outcome,
    stats: RelaxStats,
    ledger: Ledger,
    history_cost: (usize, usize, f64),
    failures: usize,
}

impl ChainReport {
    /// The chain's lowest recorded minimum.
    fn best(&self) -> Option<(f64, &Array1<f64>)> {
        self.outcome
            .best_state
            .as_ref()
            .filter(|_| self.outcome.best.is_finite())
            .map(|x| (self.outcome.best, x))
    }
}

fn main() {
    let opt = options();
    let (symbols, x0, species) = read_xyz(&opt.xyz);
    let n = symbols.len();
    let mut cfg = Config::recommended_molecular(species.clone(), vec![(0..n).collect()], 1.0);
    let fragments = connectivity_groups_z(x0.view(), &species, cfg.bond_tolerance);
    println!("molecules: {}", fragments.len());
    cfg.move_library = MoveLibrary::Molecular {
        groups: fragments.clone(),
        reactive: false,
    };
    if let Some(steps) = opt.polish {
        cfg.polish_records = steps;
    }
    if let Some(on) = opt.precondition {
        cfg.precondition_bonds = on;
    }
    if let Some(k) = opt.height_revisits {
        cfg.height_revisits = k;
    }
    let engine = std::env::var("ASE_ENGINE").unwrap_or_else(|_| "xtb-cli".into());
    println!(
        "replicas {}  history {}  identity {} A  record_gradient {:e}  polish {}  \
         precondition {}  height_revisits {}  settle {}",
        opt.replicas,
        opt.history.name(),
        opt.identity,
        cfg.record_gradient,
        cfg.polish_records,
        cfg.precondition_bonds,
        cfg.height_revisits,
        opt.settle
    );
    let reserve = opt.reserve();

    let descriptor = universal_descriptor_space(
        DescriptorGeometry::finite(cfg.length_scale).expect("descriptor geometry"),
    );
    let system = format!("{}-atoms-{}-fragments", n, fragments.len());
    let context = StructureContext::new(Some(species.clone()), None, Some(system.clone()));
    let rmsd = HeavyAtomRmsd::new(&species);
    let identity = opt.identity;
    let witness = |l: ArrayView1<f64>, r: ArrayView1<f64>| rmsd.rmsd(l, r) <= identity;
    let histories: Vec<Mutex<MinimumHistory>> = (0..match opt.history {
        History::Shared => 1,
        History::Private => opt.replicas,
        History::None => 0,
    })
        .map(|_| Mutex::new(MinimumHistory::new(cfg.record_gradient).expect("tolerance")))
        .collect();

    let share =
        |total: usize, r: usize| total / opt.replicas + usize::from(r < total % opt.replicas);
    let budgets: Vec<usize> = (0..opt.replicas)
        .map(|r| share(opt.budget - reserve, r))
        .collect();
    let seeds: Vec<u64> = (0..opt.replicas)
        .map(|r| {
            opt.seed
                .wrapping_add((r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        })
        .collect();
    let log = CallLog {
        started: Instant::now(),
        calls: AtomicUsize::new(0),
        trace: Mutex::new(BufWriter::new(
            File::create(format!("{}.trace", opt.prefix)).expect("trace"),
        )),
    };
    let turns = Turns::new(opt.replicas);
    let wall = Instant::now();
    let reports: Vec<ChainReport> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..opt.replicas)
            .map(|chain| {
                let (cfg, x0, symbols, turns, log) = (&cfg, &x0, &symbols, &turns, &log);
                let (descriptor, context, witness, engine) =
                    (&descriptor, &context, &witness, &engine);
                let history = match opt.history {
                    History::Shared => Some(&histories[0]),
                    History::Private => Some(&histories[chain]),
                    History::None => None,
                };
                let (budget, seed) = (budgets[chain], seeds[chain]);
                scope.spawn(move || {
                    let _retire = Retire(turns, chain);
                    turns.wait(chain);
                    let objective = Xtb {
                        engine: Mutex::new(PipeEngine::start(engine, symbols.clone(), None)),
                        bounds: Bounds::new(
                            Array1::from_elem(3 * n, -1.0e3),
                            Array1::from_elem(3 * n, 1.0e3),
                            0.0,
                        ),
                        chain,
                        turns,
                        log,
                    };
                    let mut hook = history.map(|history| {
                        SharedMinimumHistory::new(
                            history,
                            descriptor,
                            context.clone(),
                            witness,
                            HistoryMembership::Accepted,
                        )
                    });
                    let mut ledger = Ledger::new(budget);
                    let (outcome, stats) = search_from_with_history(
                        &objective,
                        cfg,
                        &mut ledger,
                        x0.view(),
                        seed,
                        hook.as_mut().map(|h| h as &mut dyn HistoryHook),
                    )
                    .expect("search");
                    let failures = objective.engine.lock().expect("engine").failures;
                    ChainReport {
                        seed,
                        budget,
                        outcome,
                        stats,
                        ledger,
                        history_cost: hook.as_ref().map_or((0, 0, 0.0), |h| h.cost()),
                        failures,
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a chain panicked"))
            .collect()
    });
    // Only the best of all chains is settled, so the reserve goes to the
    // minimum the run reports and not to chains that lost.
    let searched = reports
        .iter()
        .filter_map(ChainReport::best)
        .min_by(|a, b| a.0.total_cmp(&b.0));
    let mut tail = Ledger::new(reserve);
    let mut settle_failures = 0;
    let settled: Option<Settled> = searched.filter(|_| reserve > 0).and_then(|(_, x)| {
        let alone = Turns::new(1);
        let objective = Xtb {
            engine: Mutex::new(PipeEngine::start(&engine, symbols.clone(), None)),
            bounds: Bounds::new(
                Array1::from_elem(3 * n, -1.0e3),
                Array1::from_elem(3 * n, 1.0e3),
                0.0,
            ),
            chain: 0,
            turns: &alone,
            log: &log,
        };
        let settled = settle(
            &objective,
            &cfg,
            &mut tail,
            x.view(),
            SETTLE_GRADIENT,
            10 * reserve,
        );
        settle_failures = objective.engine.lock().expect("engine").failures;
        settled
    });
    let wall_s = wall.elapsed().as_secs_f64();
    log.trace.lock().expect("trace").flush().expect("trace");

    let calls = log.calls.load(Ordering::SeqCst);
    let charged: usize = reports.iter().map(|r| r.ledger.spent()).sum::<usize>() + tail.spent();
    let reported = match (searched, settled.as_ref()) {
        (Some(a), Some(b)) if b.energy < a.0 => Some((b.energy, &b.state)),
        (Some(a), _) => Some(a),
        (None, b) => b.map(|b| (b.energy, &b.state)),
    };
    let best = reported.map_or(f64::INFINITY, |(e, _)| e);
    let mut entries: Vec<(f64, Array1<f64>)> = Vec::new();
    for (chain, r) in reports.iter().enumerate() {
        let validated = r
            .ledger
            .quench_boundaries()
            .iter()
            .filter(|b| b.status() == QuenchStatus::Validated)
            .count();
        println!(
            "replica {chain} seed {} budget {} charged {} hops {} basins {} best_eV {:.8} \
             minimisations {} validated {} shared_deposits {} history_observations {} \
             history_refusals {} history_s {:.3}",
            r.seed,
            r.budget,
            r.ledger.spent(),
            r.outcome.hops,
            r.outcome.basins,
            r.outcome.best,
            r.stats.converged,
            validated,
            r.outcome.shared_deposits,
            r.history_cost.0,
            r.history_cost.1,
            r.history_cost.2,
        );
        for &(hops, at, basins, e) in &r.outcome.improvements {
            println!("improve replica={chain} hops={hops} charged={at} basins={basins} eV={e:.8}");
        }
        for b in r.ledger.quench_boundaries() {
            if b.status() == QuenchStatus::Validated && b.energy().is_finite() {
                entries.push((b.energy(), b.state().to_owned()));
            }
        }
        if let Some((energy, state)) = r.best() {
            entries.push((energy, state.clone()));
        }
    }
    println!(
        "settle reserve {reserve} charged {} settled_eV {:.8} settled_gradient {:.3e}",
        tail.spent(),
        settled.as_ref().map_or(f64::NAN, |s| s.energy),
        settled.as_ref().map_or(f64::NAN, |s| s.gradient),
    );
    if let Some(s) = settled.as_ref() {
        entries.push((s.energy, s.state.clone()));
    }
    for (k, h) in histories.iter().enumerate() {
        let h = h.lock().expect("history");
        println!(
            "history {k} minima {} accepted {} visits {}",
            h.minimum_count(),
            h.accepted_count(),
            h.total_visits()
        );
    }
    println!("best_eV {best:.8}");
    println!("best_Eh {:.10}", best / HARTREE_EV);
    println!("calls {calls}");
    println!("charged {charged}");
    println!(
        "hops {}",
        reports.iter().map(|r| r.outcome.hops).sum::<usize>()
    );
    println!(
        "basins {}",
        reports.iter().map(|r| r.outcome.basins).sum::<usize>()
    );
    println!(
        "minimisations {}",
        reports.iter().map(|r| r.stats.converged).sum::<usize>()
    );
    println!(
        "screens {}",
        reports.iter().map(|r| r.stats.screens).sum::<usize>()
    );
    println!("wall_s {wall_s:.3}");
    println!(
        "failures {}",
        reports.iter().map(|r| r.failures).sum::<usize>() + settle_failures
    );
    assert_eq!(calls, charged, "every charge is one engine call");

    let best_state = reported.map(|(_, x)| x.clone());
    let mut best_file = File::create(format!("{}.best.xyz", opt.prefix)).expect("best");
    write_frame(
        &mut best_file,
        &symbols,
        best_state.as_ref().unwrap_or(&x0).view(),
        &format!("E_eV {best:.8}"),
    );

    let corpus = MinimaCorpus::open(
        &opt.corpus,
        signature(&species, &fragments, &engine, &descriptor, cfg.length_scale).digest(),
    )
    .expect("open the minima corpus");
    let set = MinimaSet {
        system,
        temperature: cfg.temperature,
        seed: opt.seed,
    };
    let views: Vec<(f64, ArrayView1<f64>)> = entries.iter().map(|(e, x)| (*e, x.view())).collect();
    let stored = corpus
        .record(
            &set,
            &species,
            &MinimaUnits {
                length: "angstrom".into(),
                energy: "eV".into(),
            },
            &views,
            0.0,
            serde_json::json!({
                "engine": engine,
                "replicas": opt.replicas,
                "history": opt.history.name(),
                "identity_angstrom": opt.identity,
                "polish_records": cfg.polish_records,
                "precondition_bonds": cfg.precondition_bonds,
                "height_revisits": cfg.height_revisits,
                "settle_share": opt.settle,
                "settle_reserve": reserve,
                "settle_gradient": SETTLE_GRADIENT,
                "record_gradient": cfg.record_gradient,
                "budget": opt.budget,
            }),
        )
        .expect("record the minima");
    let read_back: Vec<_> = corpus
        .minima(&set.system, set.temperature)
        .expect("read the minima")
        .into_iter()
        .filter(|m| m.set.seed == opt.seed)
        .collect();
    let mut minima =
        BufWriter::new(File::create(format!("{}.minima.xyz", opt.prefix)).expect("minima"));
    for m in &read_back {
        write_frame(
            &mut minima,
            &symbols,
            ArrayView1::from(m.coordinates.as_slice()),
            &format!("E_eV {:.8} status validated", m.energy),
        );
    }
    minima.flush().expect("minima");
    println!("corpus_frames {stored}");
    println!("validated_minima {}", read_back.len());
    println!(
        "quench_records {}",
        reports
            .iter()
            .map(|r| r.ledger.quench_boundaries().len())
            .sum::<usize>()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chiral() -> (Vec<u32>, Array1<f64>) {
        let species = vec![6, 7, 8, 9, 6, 1];
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.4, 0.1, -0.2, -0.5, 1.3, 0.3, -0.4, -0.7, 1.1, -0.6, -0.8, -1.2, 2.0,
            2.0, 2.0,
        ]);
        (species, x)
    }

    fn rotated(x: &Array1<f64>, axis: [f64; 3], angle: f64, shift: [f64; 3]) -> Array1<f64> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let [u, v, w] = axis.map(|a| a / norm);
        let (s, c) = angle.sin_cos();
        let r = [
            [
                c + u * u * (1.0 - c),
                u * v * (1.0 - c) - w * s,
                u * w * (1.0 - c) + v * s,
            ],
            [
                v * u * (1.0 - c) + w * s,
                c + v * v * (1.0 - c),
                v * w * (1.0 - c) - u * s,
            ],
            [
                w * u * (1.0 - c) - v * s,
                w * v * (1.0 - c) + u * s,
                c + w * w * (1.0 - c),
            ],
        ];
        let mut out = x.clone();
        for i in 0..x.len() / 3 {
            for k in 0..3 {
                out[3 * i + k] = (0..3).map(|j| r[k][j] * x[3 * i + j]).sum::<f64>() + shift[k];
            }
        }
        out
    }

    #[test]
    fn a_rotated_and_shifted_copy_is_the_same_conformer() {
        let (species, x) = chiral();
        let rmsd = HeavyAtomRmsd::new(&species);
        let y = rotated(&x, [0.3, -1.0, 0.7], 2.1, [5.0, -3.0, 1.0]);
        assert!(rmsd.rmsd(x.view(), y.view()) < 1e-6);
    }

    #[test]
    fn a_mirror_image_is_not_superposed() {
        let (species, x) = chiral();
        let rmsd = HeavyAtomRmsd::new(&species);
        let mut mirror = x.clone();
        for i in 0..species.len() {
            mirror[3 * i + 2] = -mirror[3 * i + 2];
        }
        assert!(rmsd.rmsd(x.view(), mirror.view()) > 0.1);
    }

    #[test]
    fn hydrogens_do_not_count() {
        let (species, x) = chiral();
        let rmsd = HeavyAtomRmsd::new(&species);
        let mut moved = x.clone();
        moved[15] += 3.0;
        assert!(rmsd.rmsd(x.view(), moved.view()) < 1e-6);
    }

    #[test]
    fn one_displaced_atom_costs_at_most_its_share() {
        let (species, x) = chiral();
        let rmsd = HeavyAtomRmsd::new(&species);
        let mut moved = x.clone();
        moved[0] += 0.5;
        let heavy = 5.0_f64;
        let d = rmsd.rmsd(x.view(), moved.view());
        assert!(d > 0.0 && d <= 0.5 / heavy.sqrt() + 1e-12, "rmsd {d}");
    }

    #[test]
    fn the_turns_visit_every_live_chain_in_order() {
        assert_eq!(Turns::next_live(&[true, true, true], 2), 0);
        assert_eq!(Turns::next_live(&[true, false, true], 0), 2);
        assert_eq!(Turns::next_live(&[false, false, true], 2), 2);
    }
}
