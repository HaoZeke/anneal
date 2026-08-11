//! Persistent ASE / xtb-cli helper. No Cap'n Proto.

use ndarray::{Array1, ArrayView1};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// One child process, many charged evaluations.
pub(crate) struct PipeEngine {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    symbols: Vec<String>,
    cell: Option<[f64; 9]>,
    pub failures: usize,
}

impl Drop for PipeEngine {
    fn drop(&mut self) {
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

impl PipeEngine {
    pub(crate) fn start(engine: &str, symbols: Vec<String>, cell: Option<[f64; 9]>) -> Self {
        let helper = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/ase_objective.py");
        let omp = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".into());
        let mut child = Command::new("python3")
            .arg(helper)
            .env("ASE_ENGINE", engine)
            .env("PYTHONUNBUFFERED", "1")
            .env("OMP_NUM_THREADS", omp)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("failed to start the ASE helper");
        let stdout = child.stdout.take().expect("helper stdout");
        Self {
            child,
            reader: BufReader::new(stdout),
            symbols,
            cell,
            failures: 0,
        }
    }

    pub(crate) fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let n = x.len() / 3;
        let stdin = self.child.stdin.as_mut()?;
        let mut msg = format!("{n}\n");
        for i in 0..n {
            msg.push_str(&format!(
                "{} {:.10} {:.10} {:.10}\n",
                self.symbols[i],
                x[3 * i],
                x[3 * i + 1],
                x[3 * i + 2]
            ));
        }
        if let Some(c) = self.cell {
            msg.push_str(&format!(
                "CELL {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10}\n",
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]
            ));
        }
        msg.push_str("EVAL\n");
        stdin.write_all(msg.as_bytes()).ok()?;
        stdin.flush().ok()?;
        let mut line = String::new();
        let nread = self.reader.read_line(&mut line).ok()?;
        if nread == 0 {
            return None;
        }
        let failed = line.starts_with("FAIL");
        let energy: f64 = if failed {
            f64::INFINITY
        } else {
            line.trim().strip_prefix("E ")?.parse().ok()?
        };
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            let mut fl = String::new();
            self.reader.read_line(&mut fl).ok()?;
            let p: Vec<f64> = fl
                .split_whitespace()
                .filter_map(|v| v.parse().ok())
                .collect();
            if p.len() == 3 {
                for k in 0..3 {
                    g[3 * i + k] = -p[k];
                }
            }
        }
        let mut done = String::new();
        self.reader.read_line(&mut done).ok()?;
        if failed {
            self.failures += 1;
            return None;
        }
        Some((energy, g))
    }
}

pub(crate) fn symbol(z: u32) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        29 => "Cu",
        79 => "Au",
        _ => "X",
    }
}
