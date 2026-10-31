//! Calls to converge one molecular relaxation, L-BFGS against ANC.
//!
//! Usage: `local_opt_bench <start.xyz> [gtol eV/A] [lbfgs|anc|anc-swart|anc-bofill ...]`
//!
//! Every value-and-gradient call goes to GFN2-xTB through the `xtb`
//! executable (`ASE_ENGINE=xtb-cli`, `XTB_EXE` to override the path), one
//! process per call, and every call is counted. The gate is the largest
//! Cartesian gradient component, by default the molecular presets'
//! `record_gradient` of 5e-2 eV/A. Prints one tab-separated line per
//! optimiser: file, optimiser, calls, final energy (eV), final max |g|
//! (eV/A, from one extra uncounted call), and whether it met the gate.
//! With `LOCAL_OPT_BENCH_OUT` set to a directory, each optimiser's final
//! geometry and the energy of every counted call are written there.

#[path = "common/pipe_engine.rs"]
mod pipe_engine;

use anneal_core::methods::anc_opt::{AncOpt, AncUnits, HessianUpdate, ModelHessian};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::Array1;
use pipe_engine::PipeEngine;

const MAX_CALLS: usize = 3_000;

fn read_xyz(path: &str) -> (Vec<String>, Vec<u32>, Array1<f64>) {
    let text = std::fs::read_to_string(path).expect("readable xyz");
    let mut lines = text.lines();
    let n: usize = lines
        .next()
        .and_then(|l| l.trim().parse().ok())
        .expect("atom count");
    lines.next();
    let mut symbols = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(3 * n);
    for line in lines.take(n) {
        let mut parts = line.split_whitespace();
        symbols.push(parts.next().expect("symbol").to_string());
        x.extend(parts.take(3).map(|v| v.parse::<f64>().expect("coordinate")));
    }
    let z = symbols
        .iter()
        .map(|s| match s.as_str() {
            "H" => 1,
            "C" => 6,
            "N" => 7,
            "O" => 8,
            "S" => 16,
            other => panic!("no atomic number for {other}"),
        })
        .collect();
    (symbols, z, Array1::from(x))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: local_opt_bench <start.xyz> [gtol] [optimisers]");
    let gtol: f64 = args.get(2).map_or(5e-2, |v| v.parse().expect("gtol"));
    let names: Vec<&str> = if args.len() > 3 {
        args[3..].iter().map(String::as_str).collect()
    } else {
        vec!["lbfgs", "anc"]
    };
    let (symbols, z, x0) = read_xyz(path);
    let label = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("start");
    let out_dir = std::env::var("LOCAL_OPT_BENCH_OUT").ok();
    for name in names {
        let mut engine = PipeEngine::start("xtb-cli", symbols.clone(), None);
        let mut calls = 0usize;
        let mut trace = Vec::new();
        let mut fg = |v: ndarray::ArrayView1<f64>| {
            if calls >= MAX_CALLS {
                return None;
            }
            calls += 1;
            let out = engine.eval(v);
            trace.push(out.as_ref().map_or(f64::NAN, |(e, _)| *e));
            out
        };
        let (_, x) = match name {
            "lbfgs" => {
                let mut opt = WarmLbfgs::default();
                opt.gtol = gtol;
                let (f, x, _) = opt.minimize(x0.view(), MAX_CALLS, &mut fg);
                (f, x)
            }
            anc => {
                let mut opt = AncOpt::new(z.clone(), AncUnits::EV_ANGSTROM);
                opt.gtol = gtol;
                match anc {
                    "anc" => {}
                    "anc-swart" => opt.model = ModelHessian::Swart,
                    "anc-bofill" => opt.update = HessianUpdate::Bofill,
                    "anc-trust" => opt.trust_radius = Some(0.3),
                    other => panic!("unknown optimiser {other}"),
                }
                let (f, x, _) = opt.minimize(x0.view(), MAX_CALLS, &mut fg);
                (f, x)
            }
        };
        let (e, g) = engine.eval(x.view()).expect("final evaluation");
        let worst = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if let Some(dir) = &out_dir {
            let mut xyz = format!("{}\n{label} {name} E {e:.8} eV\n", symbols.len());
            for (i, s) in symbols.iter().enumerate() {
                xyz.push_str(&format!(
                    "{s} {:.8} {:.8} {:.8}\n",
                    x[3 * i],
                    x[3 * i + 1],
                    x[3 * i + 2]
                ));
            }
            std::fs::write(format!("{dir}/{label}-{name}.xyz"), xyz).expect("write endpoint");
            let lines: String = trace.iter().map(|e| format!("{e:.8}\n")).collect();
            std::fs::write(format!("{dir}/{label}-{name}.trace"), lines).expect("write trace");
        }
        println!(
            "{label}\t{name}\t{calls}\t{e:.6}\t{worst:.4e}\t{}",
            worst < gtol
        );
    }
}
