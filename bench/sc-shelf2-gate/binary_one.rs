use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::BinaryLennardJones;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct Timed {
    inner: BinaryLennardJones,
    elapsed_ns: AtomicU64,
}

impl Timed {
    fn measure(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let started = Instant::now();
        let value = self.inner.value_and_gradient(x);
        self.elapsed_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        value
    }
}

impl Objective<f64> for Timed {
    fn dim(&self) -> usize {
        Objective::dim(&self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.measure(x).0
    }
}

impl Gradient<f64> for Timed {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.measure(x).1
    }

    fn dim(&self) -> usize {
        Gradient::dim(&self.inner)
    }
}

impl DifferentiableObjective<f64> for Timed {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.measure(x)
    }
}

fn read_coords(path: &str) -> Array1<f64> {
    let mut coords = Vec::new();
    for line in BufReader::new(File::open(path).expect("coordinate file")).lines() {
        let line = line.expect("coordinate line");
        if line.split_whitespace().count() < 3 {
            continue;
        }
        for token in line.split_whitespace().take(3) {
            coords.push(token.parse::<f64>().expect("coordinate"));
        }
    }
    assert_eq!(coords.len(), 300, "binary start must contain 100 atoms");
    Array1::from_vec(coords)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        4,
        "binary_one <coords> <budget> <target> <seed>"
    );
    let start = read_coords(&args[0]);
    let budget: usize = args[1].parse().expect("budget");
    let target: f64 = args[2].parse().expect("target");
    let seed: u64 = args[3].parse().expect("seed");

    let mut species = vec![1u32; 42];
    species.extend(std::iter::repeat_n(2u32, 58));
    let mut sigma = vec![1.0; 42];
    sigma.extend(std::iter::repeat_n(1.3, 58));
    let config = Config::recommended(100).with_species(species, Some(sigma));
    let potential = Timed {
        inner: BinaryLennardJones::new(42, 58, 1.0, 1.0, 1.0, 1.0, 1.15, 1.3),
        elapsed_ns: AtomicU64::new(0),
    };
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(target);

    let started = Instant::now();
    let _ = search_from(&potential, &config, &mut ledger, start.view(), seed)
        .expect("finite binary start");
    let wall_s = started.elapsed().as_secs_f64();
    let pot_s = potential.elapsed_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    let outside_s = (wall_s - pot_s).max(0.0);
    let hit = ledger.best <= target;
    let calls = if hit { ledger.spent() } else { budget };
    println!(
        "h2h_row system=binary-lj code=anneal seed={seed} budget={budget} hit={} \
         evals_to_hit={calls} charged={} best={:.10} wall_s={wall_s:.6} \
         pot_s={pot_s:.6} outside_s={outside_s:.6}",
        u8::from(hit),
        ledger.spent(),
        ledger.best,
    );
}
