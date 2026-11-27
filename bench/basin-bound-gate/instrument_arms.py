#!/usr/bin/env python3
"""Install the measurement overlay in one detached arm worktree."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


MODULE = r"""use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

static RSS_ANON_KB: AtomicU64 = AtomicU64::new(0);
static VM_HWM_KB: AtomicU64 = AtomicU64::new(0);
static FIXED_RING_BOUND: AtomicU64 = AtomicU64::new(0);

fn raise(target: &AtomicU64, value: u64) {
    let mut seen = target.load(Ordering::Relaxed);
    while value > seen {
        match target.compare_exchange_weak(
            seen,
            value,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(current) => seen = current,
        }
    }
}

pub(crate) fn sample() {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return;
    };
    for line in status.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("RssAnon:") => {
                if let Some(value) = fields.next().and_then(|v| v.parse().ok()) {
                    raise(&RSS_ANON_KB, value);
                }
            }
            Some("VmHWM:") => {
                if let Some(value) = fields.next().and_then(|v| v.parse().ok()) {
                    raise(&VM_HWM_KB, value);
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn count_fixed_ring_bound() {
    FIXED_RING_BOUND.fetch_add(1, Ordering::Relaxed);
}

pub struct Sampler {
    stop: Arc<AtomicBool>,
    sampler: Option<JoinHandle<()>>,
}

impl Sampler {
    pub fn start() -> Self {
        RSS_ANON_KB.store(0, Ordering::Relaxed);
        VM_HWM_KB.store(0, Ordering::Relaxed);
        FIXED_RING_BOUND.store(0, Ordering::Relaxed);
        sample();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let sampler = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                sample();
                thread::sleep(Duration::from_millis(10));
            }
            sample();
        });
        Self {
            stop,
            sampler: Some(sampler),
        }
    }

    fn stop_and_report(&mut self, report: bool) {
        sample();
        self.stop.store(true, Ordering::Relaxed);
        if let Some(sampler) = self.sampler.take() {
            let _ = sampler.join();
        }
        sample();
        if report {
            eprintln!(
                "gate_memory rss_anon_kb={} vm_hwm_kb={} fixed_ring_bound={}",
                RSS_ANON_KB.load(Ordering::Relaxed),
                VM_HWM_KB.load(Ordering::Relaxed),
                FIXED_RING_BOUND.load(Ordering::Relaxed),
            );
        }
    }

    pub fn finish(mut self) {
        self.stop_and_report(true);
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop_and_report(false);
    }
}
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_durable(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    if sha256(path.read_bytes()) != sha256(data):
        raise RuntimeError(f"read-back mismatch for {path.name}")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path.name}: expected one overlay marker")
    write_durable(path, text.replace(old, new, 1).encode("utf-8"))


def wrap_search(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("    let (outcome, wall, pot_s) = search(")
    ]
    if len(starts) != 1 or not lines[starts[0]].rstrip().endswith(";"):
        raise RuntimeError(f"{path.name}: timed search marker missing")
    index = starts[0]
    lines.insert(index, "    let gate_sampler = anneal_core::gate_memory::Sampler::start();\n")
    lines.insert(index + 2, "    gate_sampler.finish();\n")
    write_durable(path, "".join(lines).encode("utf-8"))


def install(root: Path, arm: str) -> None:
    driver_source = Path(__file__).resolve().parent / "driver-src"
    for name in ("h2h_blj.rs", "h2h_morse.rs"):
        write_durable(root / "examples" / name, (driver_source / name).read_bytes())
    write_durable(root / "src" / "gate_memory.rs", MODULE.encode("utf-8"))
    replace_once(
        root / "src" / "lib.rs",
        "pub mod bias;\n",
        "pub mod bias;\n#[doc(hidden)]\npub mod gate_memory;\n",
    )
    replace_once(
        root / "src" / "bias.rs",
        "        self.visits.push(0);\n        self.centres.len() - 1\n",
        "        self.visits.push(0);\n"
        "        crate::gate_memory::sample();\n"
        "        self.centres.len() - 1\n",
    )
    for name in ("h2h_gmin_proof.rs", "h2h_blj.rs", "h2h_morse.rs", "h2h_sc.rs"):
        wrap_search(root / "examples" / name)
    replace_once(
        root / "examples" / "h2h_seed.rs",
        "    let started = Instant::now();\n",
        "    let gate_sampler = anneal_core::gate_memory::Sampler::start();\n"
        "    let started = Instant::now();\n",
    )
    replace_once(
        root / "examples" / "h2h_seed.rs",
        "    let wall = started.elapsed().as_secs_f64();\n",
        "    let wall = started.elapsed().as_secs_f64();\n"
        "    gate_sampler.finish();\n",
    )
    if arm == "B":
        bias = root / "src" / "bias.rs"
        text = bias.read_text(encoding="utf-8")
        marker = (
            "        let bounded = self.metric.is_euclidean() "
            "&& aggregate_rounding_is_small(d.len());\n"
        )
        if text.count(marker) != 1:
            raise RuntimeError("bias.rs: fixed-ring bound marker missing")
        text = text.replace(
            marker,
            marker
            + "        if bounded {\n"
            + "            crate::gate_memory::count_fixed_ring_bound();\n"
            + "        }\n",
            1,
        )
        write_durable(bias, text.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "A2", "B"), required=True)
    args = parser.parse_args()
    install(args.root.resolve(), args.arm)
    print(sha256(MODULE.encode("utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
