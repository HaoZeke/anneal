#!/usr/bin/env python3
"""Add measurement-only instrumentation to one benchmark source tree."""

from __future__ import annotations

import argparse
from pathlib import Path


MEMORY_CODE = r'''
#[derive(Clone, Copy)]
struct GateMemory {
    rss_anon_peak_kb: usize,
    vm_hwm_kb: usize,
    ru_maxrss_kb: usize,
}

struct GateMemorySampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn gate_status_kb() -> (usize, usize) {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let mut rss_anon = 0;
    let mut vm_hwm = 0;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("RssAnon:") => {
                rss_anon = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            Some("VmHWM:") => {
                vm_hwm = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            _ => {}
        }
    }
    (rss_anon, vm_hwm)
}

#[cfg(target_os = "linux")]
fn gate_ru_maxrss_kb() -> usize {
    use std::ffi::{c_int, c_long};

    #[repr(C)]
    struct TimeVal {
        seconds: c_long,
        microseconds: c_long,
    }

    #[repr(C)]
    struct ResourceUsage {
        user: TimeVal,
        system: TimeVal,
        max_rss: c_long,
        rest: [c_long; 13],
    }

    unsafe extern "C" {
        fn getrusage(who: c_int, usage: *mut ResourceUsage) -> c_int;
    }

    let mut usage = std::mem::MaybeUninit::<ResourceUsage>::uninit();
    if unsafe { getrusage(0, usage.as_mut_ptr()) } == 0 {
        unsafe { usage.assume_init() }.max_rss.max(0) as usize
    } else {
        0
    }
}

#[cfg(not(target_os = "linux"))]
fn gate_ru_maxrss_kb() -> usize {
    0
}

impl GateMemorySampler {
    fn start() -> Self {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let thread_stop = stop.clone();
        let thread_peak = peak.clone();
        let thread = std::thread::spawn(move || {
            while !thread_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let (rss_anon, _) = gate_status_kb();
                thread_peak.fetch_max(rss_anon, std::sync::atomic::Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            let (rss_anon, _) = gate_status_kb();
            thread_peak.fetch_max(rss_anon, std::sync::atomic::Ordering::Relaxed);
        });
        Self {
            stop,
            peak,
            thread: Some(thread),
        }
    }

    fn finish(mut self) -> GateMemory {
        self.stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let (rss_anon, vm_hwm) = gate_status_kb();
        GateMemory {
            rss_anon_peak_kb: self
                .peak
                .load(std::sync::atomic::Ordering::Relaxed)
                .max(rss_anon),
            vm_hwm_kb: vm_hwm,
            ru_maxrss_kb: gate_ru_maxrss_kb(),
        }
    }
}
'''


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected one occurrence, found {text.count(old)}: {old[:60]!r}")
    return text.replace(old, new, 1)


def instrument(source: Path, layout: int, generic: bool) -> None:
    hopping = source / "src/methods/cluster_hopping.rs"
    text = hopping.read_text()
    text = replace_once(text, "\nfn run_full<'g, R, H>(", MEMORY_CODE + "\nfn run_full<'g, R, H>(")
    marker = ") -> Result<Outcome, SearchError>\nwhere\n"
    start = text.index("fn run_full<'g, R, H>(")
    signature = text.index(marker, start)
    body = text.index("{", signature + len(marker))
    text = (
        text[: body + 1]
        + "\n    let gate_memory = GateMemorySampler::start();"
        + "\n    let mut gate_packing_offered = false;"
        + "\n    let mut gate_packing_hit = false;"
        + "\n    let mut gate_packing_probe_calls = 0usize;\n"
        + text[body + 1 :]
    )
    if generic:
        text = replace_once(
            text,
            "            packings_offered = true;\n",
            "            packings_offered = true;\n            gate_packing_offered = true;\n",
        )
        text = replace_once(
            text,
            "            let probe_spent = probe.spent().saturating_sub(probe_offset);\n",
            "            let probe_spent = probe.spent().saturating_sub(probe_offset);\n"
            "            gate_packing_probe_calls = probe_spent;\n",
        )
        text = replace_once(
            text,
            "                    {\n                        ledger.record(cut_energy, cut_state.view());\n",
            "                    {\n                        gate_packing_hit = true;\n"
            "                        ledger.record(cut_energy, cut_state.view());\n",
        )
    outcome = text.rfind("    Ok(Outcome {\n")
    if outcome < 0:
        raise SystemExit("final outcome not found")
    report = """    let gate_memory = gate_memory.finish();
    eprintln!(
        "R3_GATE_METRICS,rss_anon_self_peak_kb={},vm_hwm_kb={},ru_maxrss_kb={},packing_offered={},packing_hit={},packing_probe_calls={}",
        gate_memory.rss_anon_peak_kb,
        gate_memory.vm_hwm_kb,
        gate_memory.ru_maxrss_kb,
        usize::from(gate_packing_offered),
        usize::from(gate_packing_hit),
        gate_packing_probe_calls,
    );
"""
    text = text[:outcome] + report + text[outcome:]
    hopping.write_text(text, newline="\n")

    layout_source = source / "src/r3_gate_layout.rs"
    layout_source.write_text(
        f"""#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn r3_gate_layout_pad() {{
    unsafe {{
        std::arch::asm!(".rept {layout}", "nop", ".endr");
    }}
}}

#[used]
static R3_GATE_LAYOUT_KEEP: extern "C" fn() = r3_gate_layout_pad;
""",
        newline="\n",
    )
    library = source / "src/lib.rs"
    library_text = library.read_text()
    library_text = replace_once(
        library_text,
        "//! consumed by the Python `anneal` package via pyo3.\n",
        "//! consumed by the Python `anneal` package via pyo3.\n"
        "\n#[doc(hidden)]\nmod r3_gate_layout;\n",
    )
    library.write_text(library_text, newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--generic", action="store_true")
    args = parser.parse_args()
    instrument(args.source, args.layout, args.generic)


if __name__ == "__main__":
    main()
