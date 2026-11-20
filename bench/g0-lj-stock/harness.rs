// Copyright (c) anneal developers
//! One seed of the stock cluster preset on the paper Lennard-Jones panel.
//!
//! The preset is `Config::recommended` with no prior flag set. Gold keeps
//! the metal driver's species list and basin stride of 8. Morse keeps the
//! Morse driver as it stands, which does not set `long_ranged`.
//!
//! A hit is a finite best energy at or below the reference plus 0.001.
//! The walk stops at that level or at the paper budget. Starts are read
//! from the committed bit file so a later draw cannot replace them.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anneal_core::continuous_symmetry::inversion_rms;
use anneal_core::corekey::contact_cutoff;
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{Gupta, PairPotential};
use anneal_core::structure::cna;
use anneal_core::symmetrise::{Candidate, ShellCore, detect_all};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use readcon_core::helpers::atomic_number_to_symbol;
use readcon_core::types::{ConFrameBuilder, meta};
use readcon_core::writer::ConFrameWriter;
use sha2::{Digest, Sha256};

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;
const SEEDS: [u64; 12] = [
    2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
];
/// Detection tolerance per length, the value `ShellCore::scaled` uses.
const POINT_GROUP_TOLERANCE_PER_LENGTH: f64 = 0.1;
/// Cosine above which two detected axes are one axis.
const AXIS_PARALLEL_COSINE: f64 = 0.995;
/// Rotation orders `detect_all` is given. The same orders the shell-core
/// reader tries.
const ROTATION_ORDERS: [usize; 5] = [2, 3, 4, 5, 6];
/// Share of 555 bonds required, with 421 below `ICO_F421_MAX`, for an
/// icosahedral call.
const ICO_F555: f64 = 0.40;
const ICO_F421_MAX: f64 = 0.10;
/// 555 and 421 shares for a decahedral call.
const DECA_F555: f64 = 0.05;
const DECA_F421: f64 = 0.10;
/// 555 ceiling and close-packed share for an fcc/twin call.
const FCC_F555_MAX: f64 = 0.05;
const FCC_CLOSE_PACKED: f64 = 0.25;
const OPEN_BOX: f64 = 500.0;

#[derive(Clone, Copy)]
enum Kind {
    LennardJones,
    Morse { rho: f64 },
    Gold,
}

struct Cell {
    name: &'static str,
    panel: &'static str,
    kind: Kind,
    n: usize,
    budget: usize,
    reference: f64,
}

fn cells() -> Vec<Cell> {
    let mut out = Vec::new();
    let lj = [
        ("lj38", 38usize, 400_000usize, -173.928427),
        ("lj55", 55, 1_000_000, -279.248470),
        ("lj75", 75, 4_000_000, -397.492331),
        ("lj98", 98, 4_000_000, -543.665361),
        ("lj102", 102, 4_000_000, -569.363652),
        ("lj103", 103, 4_000_000, -575.766131),
        ("lj104", 104, 4_000_000, -582.086642),
        ("lj150", 150, 4_000_000, -893.310258),
    ];
    for (system, n, budget, reference) in lj {
        out.push(Cell {
            name: leak(format!("gmin-{system}")),
            panel: "gmin",
            kind: Kind::LennardJones,
            n,
            budget,
            reference,
        });
    }
    out.push(Cell {
        name: "gmin-au55",
        panel: "gmin",
        kind: Kind::Gold,
        n: 55,
        budget: 1_000_000,
        reference: -191.03085280,
    });
    let morse = [
        ("m55r6", 55usize, 6.0, 1_000_000usize, -250.286609),
        ("m55r14", 55, 14.0, 1_000_000, -220.646208),
        ("m80r6", 80, 6.0, 4_000_000, -378.333471),
        ("m80r14", 80, 14.0, 4_000_000, -340.811371),
    ];
    for (system, n, rho, budget, reference) in morse {
        out.push(Cell {
            name: leak(format!("gmin-{system}")),
            panel: "gmin",
            kind: Kind::Morse { rho },
            n,
            budget,
            reference,
        });
    }
    let ase = [
        ("lj38", 38usize, 400_000usize, -173.928427),
        ("lj55", 55, 1_000_000, -279.248470),
        ("lj98", 98, 4_000_000, -543.665361),
        ("lj104", 104, 4_000_000, -582.086642),
        ("lj150", 150, 4_000_000, -893.310258),
    ];
    for (system, n, budget, reference) in ase {
        out.push(Cell {
            name: leak(format!("ase-{system}")),
            panel: "ase",
            kind: Kind::LennardJones,
            n,
            budget,
            reference,
        });
    }
    out
}

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn find_cell(name: &str) -> Option<Cell> {
    cells().into_iter().find(|cell| cell.name == name)
}

fn stock(cell: &Cell) -> Config {
    let mut cfg = Config::recommended(cell.n);
    if matches!(cell.kind, Kind::Gold) {
        let species = Gupta::copper_gold_species(0, cell.n);
        cfg = cfg.with_species(species, None);
        cfg.basin_stride = 8;
    }
    cfg
}

fn draw_start(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn start_digest(coords: &Array1<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in coords.iter() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hex_bytes(&hasher.finalize())
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_start(path: &Path, coords: &Array1<f64>) -> std::io::Result<()> {
    let mut text = String::from("bits\n");
    for value in coords.iter() {
        text.push_str(&format!("{:016x}\n", value.to_bits()));
    }
    let mut file = File::create(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

fn read_start(path: &Path, n: usize) -> Result<Array1<f64>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or_else(|| "start file is empty".to_string())?
        .map_err(|err| err.to_string())?;
    if header.trim() != "bits" {
        return Err("start file is not a bit list".into());
    }
    let mut values = Vec::with_capacity(3 * n);
    for line in lines {
        let line = line.map_err(|err| err.to_string())?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let bits = u64::from_str_radix(text, 16).map_err(|err| err.to_string())?;
        values.push(f64::from_bits(bits));
    }
    if values.len() != 3 * n {
        return Err(format!(
            "start has {} values, wanted {}",
            values.len(),
            3 * n
        ));
    }
    Ok(Array1::from_vec(values))
}

fn machine_id() -> String {
    let text = std::fs::read_to_string("/etc/machine-id").unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(text.trim().as_bytes());
    hex_bytes(&hasher.finalize())
}

fn core_and_affinity() -> (String, String) {
    let list = affinity();
    let core = list
        .split([',', '-'])
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    (core, list)
}

fn affinity() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("Cpus_allowed_list"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_string())
        })
        .unwrap_or_default()
}

fn revision() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_default()
}

fn refuse_h2h() -> Result<(), String> {
    Config::check_removed_env().map_err(|err| err.to_string())?;
    for (key, _) in std::env::vars() {
        if key.starts_with("H2H_") {
            return Err(format!("refusing environment variable {key}"));
        }
    }
    Ok(())
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    if anneal_core::potentials::cluster_kernels().is_none() {
        eprintln!("pair kernels are not installed");
        std::process::exit(1);
    }
    if (HIT_TOL - 0.001).abs() > 1e-15 {
        eprintln!("hit tolerance is not 0.001");
        std::process::exit(1);
    }
    if let Err(err) = refuse_h2h() {
        eprintln!("{err}");
        std::process::exit(2);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let status = match cmd {
        "list" => cmd_list(),
        "knobs" => cmd_knobs(),
        "coords-all" => cmd_coords_all(args.get(1).map(String::as_str)),
        "run" => cmd_run(&args),
        _ => {
            eprintln!(
                "usage: list | knobs | coords-all DIR | run CELL SEED START SESSION SEQUENCE MISS_DIR"
            );
            2
        }
    };
    std::process::exit(status);
}

fn cmd_list() -> i32 {
    println!("cell,panel,n,budget,reference,kind");
    for cell in cells() {
        let kind = match cell.kind {
            Kind::LennardJones => "lj".to_string(),
            Kind::Morse { rho } => format!("morse-{rho}"),
            Kind::Gold => "gold".to_string(),
        };
        println!(
            "{},{},{},{},{:.10},{kind}",
            cell.name, cell.panel, cell.n, cell.budget, cell.reference
        );
    }
    0
}

fn cmd_knobs() -> i32 {
    for cell in cells() {
        let cfg = stock(&cell);
        println!("BEGIN {}", cell.name);
        println!("long_ranged={}", cfg.long_ranged);
        println!("restart_on_stall={}", cfg.restart_on_stall);
        println!("escape_on_stall={}", cfg.escape_on_stall);
        println!("energy_lattice={}", cfg.energy_lattice);
        println!("relax_steps={}", cfg.relax_steps);
        println!("basin_stride={}", cfg.basin_stride);
        println!("orbit_complete_on_new={}", cfg.orbit_complete_on_new);
        println!("symmetrise_on_stall={}", cfg.symmetrise_on_stall);
        println!("return_screen={}", cfg.return_screen);
        println!("tabu_on_stall={}", cfg.tabu_on_stall);
        println!("trail_on_stall={}", cfg.trail_on_stall);
        println!("allocate_moves={}", cfg.allocate_moves);
        println!("depth_reward={}", cfg.depth_reward);
        println!("bayes_screen={}", cfg.bayes_screen);
        println!("prior_soap={}", cfg.prior_soap);
        println!("surfaces={}", cfg.surfaces.len());
        println!("container={:.16}", cfg.container);
        println!("min_separation={:.16}", cfg.min_separation);
        println!("length_scale={:.16}", cfg.length_scale);
        println!("temperature={:.16}", cfg.temperature);
        println!("move_library={:?}", cfg.move_library);
        println!("orbit_core={:?}", cfg.orbit_core);
        println!("continuous_symmetry={:?}", cfg.continuous_symmetry);
        println!("soap_mode={:?}", cfg.soap_mode);
        println!("keying={:?}", cfg.keying);
        println!("species={}", cfg.species.is_some());
        if let Kind::Morse { rho } = cell.kind {
            let second = anneal_core::potentials::PairKind::Morse { rho }.second_neighbour_energy();
            println!("second_neighbour={second:.16}");
        }
        match cfg.resolved_sha256() {
            Ok(hash) => println!("sha256={hash}"),
            Err(err) => {
                eprintln!("{err}");
                return 1;
            }
        }
        println!("DEBUG");
        println!("{cfg:#?}");
        println!("END {}", cell.name);
    }
    0
}

fn cmd_coords_all(dir: Option<&str>) -> i32 {
    let Some(dir) = dir else {
        eprintln!("coords-all needs a directory");
        return 2;
    };
    if let Err(err) = std::fs::create_dir_all(dir) {
        eprintln!("{err}");
        return 1;
    }
    println!("cell,seed,start");
    for cell in cells() {
        let cfg = stock(&cell);
        for seed in SEEDS {
            let coords = draw_start(&cfg, cell.n, seed);
            if coords.len() != 3 * cell.n {
                eprintln!("{} {seed}: start placed {} values", cell.name, coords.len());
                return 1;
            }
            let path = Path::new(dir).join(format!("{}-{seed}.txt", cell.name));
            if let Err(err) = write_start(&path, &coords) {
                eprintln!("{err}");
                return 1;
            }
            println!("{},{seed},{}", cell.name, start_digest(&coords));
        }
    }
    0
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(cell) = args.get(1).and_then(|name| find_cell(name)) else {
        eprintln!("run needs a known cell");
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    if !SEEDS.contains(&seed) {
        eprintln!("seed is outside 2000-2011");
        return 2;
    }
    let Some(start_path) = args.get(3) else {
        eprintln!("run needs a start file");
        return 2;
    };
    let Some(session) = args.get(4) else {
        eprintln!("run needs a session");
        return 2;
    };
    let Some(sequence) = args.get(5) else {
        eprintln!("run needs a sequence");
        return 2;
    };
    let Some(miss_dir) = args.get(6) else {
        eprintln!("run needs a miss directory");
        return 2;
    };
    let start = match read_start(Path::new(start_path), cell.n) {
        Ok(start) => start,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let cfg = stock(&cell);
    let hash = match cfg.resolved_sha256() {
        Ok(hash) => hash,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let (outcome, wall, pot_s) = match search(&cell, &cfg, &start, seed) {
        Ok(row) => row,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let stop = outcome.stop.as_ref().map(|stop| stop.code()).unwrap_or("");
    if stop != "budget" && stop != "target" {
        eprintln!("search stopped at {stop}");
        return 1;
    }
    let best = outcome.best;
    let hit = best.is_finite() && anneal_core::energy_hits(best, cell.reference, HIT_TOL);
    if hit && stop != "target" && stop != "budget" {
        eprintln!("hit without a finished stop");
        return 1;
    }
    let calls = calls_to_hit(&outcome, cell.reference, hit);
    let outside = (wall - pot_s).max(0.0);
    let hop = hop_first_best(&outcome);
    let bits = if best.is_finite() {
        format!("{:016x}", best.to_bits())
    } else {
        String::new()
    };
    let calls_text = calls.map(|value| value.to_string()).unwrap_or_default();
    let (core, pins) = core_and_affinity();
    if !hit {
        let path = Path::new(miss_dir).join(format!("{}-s{seed}.con", cell.name));
        if let Err(err) = write_miss(&cell, &outcome, &path) {
            eprintln!("{err}");
            return 1;
        }
        let Some(state) = outcome.best_state.as_ref() else {
            eprintln!("miss has no structure");
            return 1;
        };
        let class = classify(&cfg, state.view());
        println!(
            "miss,{},{seed},{},{:.8},{},{},{:.8}",
            cell.name,
            class.point_group,
            class.tolerance,
            class.signature,
            class.kind,
            class.cutoff
        );
    }
    println!(
        "row,{name},{seed},{start},{hit},{calls},{charged},{best:.16},{bits},{wall:.6},{pot:.6},{outside:.6},{hop},{hash},{rev},{core},{pins},{machine},{session},{sequence}",
        name = cell.name,
        start = start_digest(&start),
        hit = u8::from(hit),
        calls = calls_text,
        charged = outcome.charged,
        best = best,
        bits = bits,
        wall = wall,
        pot = pot_s,
        outside = outside,
        hop = hop,
        hash = hash,
        rev = revision(),
        core = core,
        pins = pins,
        machine = machine_id(),
        session = session,
        sequence = sequence,
    );
    0
}

fn calls_to_hit(outcome: &Outcome, reference: f64, hit: bool) -> Option<usize> {
    if !hit {
        return None;
    }
    let level = reference + HIT_TOL;
    outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| energy.is_finite() && *energy <= level)
        .map(|(_, spent, _, _)| *spent)
        .or(Some(outcome.charged))
}

fn hop_first_best(outcome: &Outcome) -> String {
    let best = outcome.best;
    if !best.is_finite() {
        return String::new();
    }
    let bits = best.to_bits();
    if let Some((hop, _, _, _)) = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| energy.to_bits() == bits)
    {
        return hop.to_string();
    }
    outcome
        .improvements
        .last()
        .map(|(hop, _, _, _)| hop.to_string())
        .unwrap_or_else(|| outcome.hops.to_string())
}

fn search(
    cell: &Cell,
    cfg: &Config,
    start: &Array1<f64>,
    seed: u64,
) -> Result<(Outcome, f64, f64), String> {
    let n = cell.n;
    let budget = cell.budget;
    let level = cell.reference + HIT_TOL;
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(level);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = AtomicU64::new(0);
    let started = Instant::now();
    let outcome = match cell.kind {
        Kind::LennardJones => {
            let potential = PairPotential::lennard_jones(n);
            finish(
                cfg,
                start,
                seed,
                &mut ledger,
                &mut optimiser,
                &pot_ns,
                started,
                move |v| potential.value_and_gradient(v),
            )
        }
        Kind::Morse { rho } => {
            let potential = PairPotential::morse(n, rho);
            finish(
                cfg,
                start,
                seed,
                &mut ledger,
                &mut optimiser,
                &pot_ns,
                started,
                move |v| potential.value_and_gradient(v),
            )
        }
        Kind::Gold => {
            let potential = Gupta::copper_gold(0, n);
            finish(
                cfg,
                start,
                seed,
                &mut ledger,
                &mut optimiser,
                &pot_ns,
                started,
                move |v| potential.value_and_gradient(v),
            )
        }
    }?;
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let wall = started.elapsed().as_secs_f64();
    Ok((outcome, wall, pot_s))
}

fn finish<F>(
    cfg: &Config,
    start: &Array1<f64>,
    seed: u64,
    ledger: &mut Ledger,
    optimiser: &mut WarmLbfgs,
    pot_ns: &AtomicU64,
    _started: Instant,
    potential: F,
) -> Result<Outcome, String>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Sync,
{
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let evaluated = potential(v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Some(evaluated)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = potential(v).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(gradient)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    run_with_gradient(
        cfg,
        start.view(),
        ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .map_err(|err| err.to_string())
}

fn write_miss(cell: &Cell, outcome: &Outcome, path: &Path) -> Result<(), String> {
    let state = outcome
        .best_state
        .as_ref()
        .ok_or_else(|| "miss has no structure".to_string())?;
    if state.len() != 3 * cell.n {
        return Err("miss structure has the wrong length".into());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let z = match cell.kind {
        Kind::Gold => 79,
        Kind::LennardJones | Kind::Morse { .. } => 0,
    };
    let symbol = atomic_number_to_symbol(z);
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        meta::UNITS.into(),
        serde_json::json!({"length": "reduced", "energy": "reduced"}),
    );
    metadata.insert(meta::PBC.into(), serde_json::json!([false, false, false]));
    metadata.insert(
        "energy_bits".into(),
        serde_json::json!(format!("{:016x}", outcome.best.to_bits())),
    );
    let mut builder = ConFrameBuilder::new([OPEN_BOX; 3], [90.0; 3]);
    builder.prebox_header(format!("{} units reduced", cell.name));
    builder.metadata(metadata);
    builder.set_energy(outcome.best);
    for atom in 0..cell.n {
        builder.add_atom(
            symbol,
            state[3 * atom],
            state[3 * atom + 1],
            state[3 * atom + 2],
            [false; 3],
            atom as u64,
            1.0,
        );
    }
    let mut frame = builder.build().map_err(|err| err.to_string())?;
    // Spec 3 replaces a unit name it cannot parse. Reduced units are the
    // native units of these potentials, so the frame stays on spec 2 and
    // the writer keeps the name.
    frame.header.spec_version = 2;
    {
        let mut writer =
            ConFrameWriter::from_path_with_precision(path, 17).map_err(|err| err.to_string())?;
        writer.write_frame(&frame).map_err(|err| err.to_string())?;
    }
    let file = File::options()
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    file.sync_all().map_err(|err| err.to_string())?;
    Ok(())
}

struct Class {
    point_group: String,
    tolerance: f64,
    signature: String,
    kind: String,
    cutoff: f64,
}

fn classify(cfg: &Config, x: ArrayView1<f64>) -> Class {
    let n = x.len() / 3;
    let tolerance = ShellCore::scaled(cfg.length_scale).tolerance;
    let expected = POINT_GROUP_TOLERANCE_PER_LENGTH * cfg.length_scale;
    assert!(
        (tolerance - expected).abs() <= 1e-12,
        "point-group tolerance is not 0.1 lengths"
    );
    let found = detect_all(x, n, &ROTATION_ORDERS, tolerance);
    let n5 = distinct(&found, 5, false);
    let n4 = distinct(&found, 4, false);
    let n3 = distinct(&found, 3, false);
    let n2 = distinct(&found, 2, false);
    let mirrors = distinct(&found, 1, true);
    let classes = vec![0u32; n];
    let inversion = inversion_rms(x, &classes).is_some_and(|rms| rms < tolerance);
    let point_group = point_group_name(n5, n4, n3, n2, mirrors, inversion);
    let cutoff = contact_cutoff(x);
    let counts = cna(x, n, cutoff);
    let f555 = counts.fraction((5, 5, 5));
    let f421 = counts.fraction((4, 2, 1));
    let f422 = counts.fraction((4, 2, 2));
    let kind = structure_kind(f555, f421, f422);
    let mut parts: Vec<String> = counts
        .counts
        .iter()
        .map(|((r, s, t), count)| format!("{r}{s}{t}:{count}"))
        .collect();
    parts.sort();
    let signature = if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(";")
    };
    Class {
        point_group,
        tolerance,
        signature,
        kind,
        cutoff,
    }
}

fn distinct(found: &[Candidate], order: usize, improper: bool) -> usize {
    let mut axes: Vec<[f64; 3]> = Vec::new();
    for candidate in found {
        if candidate.order != order || candidate.improper != improper {
            continue;
        }
        if axes.iter().any(|axis| same_axis(*axis, candidate.axis)) {
            continue;
        }
        axes.push(candidate.axis);
    }
    axes.len()
}

fn same_axis(left: [f64; 3], right: [f64; 3]) -> bool {
    let dot = left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    dot.abs() > AXIS_PARALLEL_COSINE
}

fn point_group_name(
    n5: usize,
    n4: usize,
    n3: usize,
    n2: usize,
    mirrors: usize,
    inversion: bool,
) -> String {
    if n5 >= 6 {
        return if mirrors > 0 || inversion { "Ih" } else { "I" }.to_string();
    }
    if n5 == 1 {
        return if mirrors > 0 { "D5h" } else { "D5" }.to_string();
    }
    if n4 >= 3 {
        return if mirrors > 0 || inversion { "Oh" } else { "O" }.to_string();
    }
    if n3 >= 4 {
        return if mirrors > 0 { "Td" } else { "T" }.to_string();
    }
    if n3 == 1 && n2 >= 3 {
        return if mirrors > 0 { "D3h" } else { "D3" }.to_string();
    }
    if n3 == 1 && mirrors > 0 {
        return "C3v".to_string();
    }
    if n3 == 1 {
        return "C3".to_string();
    }
    if n2 >= 1 && mirrors > 0 {
        return "C2v".to_string();
    }
    if n2 >= 1 {
        return "C2".to_string();
    }
    if mirrors > 0 {
        return "Cs".to_string();
    }
    if inversion {
        return "Ci".to_string();
    }
    "C1".to_string()
}

fn structure_kind(f555: f64, f421: f64, f422: f64) -> String {
    if f555 >= ICO_F555 && f421 < ICO_F421_MAX {
        "icosahedral".to_string()
    } else if f555 >= DECA_F555 && f421 >= DECA_F421 {
        "decahedral".to_string()
    } else if f555 < FCC_F555_MAX && (f421 + f422) >= FCC_CLOSE_PACKED {
        "fcc/twin".to_string()
    } else {
        "disordered".to_string()
    }
}
