// Copyright (c) anneal developers
//! Shared helpers for the r-next2 gate drivers.

use std::fs;

#[cfg(feature = "placement-pad")]
#[used]
static PLACEMENT_PAD: [u8; 65536] = [0xA5; 65536];

#[inline(never)]
pub fn placement_pad_byte() -> u8 {
    #[cfg(feature = "placement-pad")]
    {
        std::hint::black_box(PLACEMENT_PAD[0])
    }
    #[cfg(not(feature = "placement-pad"))]
    {
        0
    }
}

pub struct ProcMem {
    pub rss_anon_kb: u64,
    pub vm_hwm_kb: u64,
    pub cpus_allowed: String,
}

pub fn read_proc_self_status() -> ProcMem {
    let mut rss_anon_kb = 0_u64;
    let mut vm_hwm_kb = 0_u64;
    let mut cpus_allowed = String::from("unknown");
    if let Ok(text) = fs::read_to_string("/proc/self/status") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("RssAnon:") {
                if let Some(num) = rest.split_whitespace().next() {
                    rss_anon_kb = num.parse().unwrap_or(0);
                }
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(num) = rest.split_whitespace().next() {
                    vm_hwm_kb = num.parse().unwrap_or(0);
                }
            } else if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
                cpus_allowed = rest.trim().to_string();
            }
        }
    }
    ProcMem {
        rss_anon_kb,
        vm_hwm_kb,
        cpus_allowed,
    }
}

pub fn print_h2h_mem(best: f64) {
    let mem = read_proc_self_status();
    println!(
        "h2h_mem rss_anon_kb={} vm_hwm_kb={} cpus={} best_bits=0x{:016x}",
        mem.rss_anon_kb,
        mem.vm_hwm_kb,
        mem.cpus_allowed,
        best.to_bits(),
    );
}
