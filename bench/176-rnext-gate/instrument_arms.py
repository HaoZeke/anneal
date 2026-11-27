#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Apply in-process RssAnon and 176-rnext path-counter instrumentation to A/B worktrees."""

import hashlib
import os
import pathlib
import sys

TELEMETRY_MOD_TEMPLATE = """
pub mod gate_telemetry {{
    use std::io::Write;
    use std::sync::Once;
    use std::sync::atomic::{{AtomicU64, Ordering}};

    const PAD_BYTES: usize = {pad_bytes};
    #[used]
    static GATE_PAD: [u8; PAD_BYTES] = [0x5au8; PAD_BYTES];

    static INIT: Once = Once::new();
    static PEAK_RSS_ANON_KB: AtomicU64 = AtomicU64::new(0);
    static PEAK_VM_HWM_KB: AtomicU64 = AtomicU64::new(0);
    static BASIS_WALKS: AtomicU64 = AtomicU64::new(0);
    static BASIS_EVIDENCE: AtomicU64 = AtomicU64::new(0);
    static BASIS_CLAIMS: AtomicU64 = AtomicU64::new(0);
    static BASIS_PROBES: AtomicU64 = AtomicU64::new(0);
    static BASIS_CANDIDATES: AtomicU64 = AtomicU64::new(0);
    static BASIS_SPENT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" {{
        fn atexit(cb: extern "C" fn()) -> i32;
    }}

    extern "C" fn on_exit() {{
        sample();
        let rss = PEAK_RSS_ANON_KB.load(Ordering::Relaxed);
        let hwm = PEAK_VM_HWM_KB.load(Ordering::Relaxed);
        let walks = BASIS_WALKS.load(Ordering::Relaxed);
        let evidence = BASIS_EVIDENCE.load(Ordering::Relaxed);
        let claims = BASIS_CLAIMS.load(Ordering::Relaxed);
        let probes = BASIS_PROBES.load(Ordering::Relaxed);
        let candidates = BASIS_CANDIDATES.load(Ordering::Relaxed);
        let spent = BASIS_SPENT.load(Ordering::Relaxed);
        let mut out = std::io::stdout().lock();
        let _ = writeln!(
            out,
            "gate_telemetry rss_anon_kb={{rss}} vm_hwm_kb={{hwm}} basis_walks={{walks}} basis_evidence={{evidence}} basis_claims={{claims}} basis_probes={{probes}} basis_candidates={{candidates}} basis_spent={{spent}} pad_bytes={{pad}}",
            pad = PAD_BYTES,
        );
        let _ = out.flush();
    }}

    #[inline]
    fn touch_pad() {{
        let ptr = GATE_PAD.as_ptr();
        let mut offset = 0usize;
        while offset < PAD_BYTES {{
            unsafe {{
                std::hint::black_box(ptr.add(offset).read_volatile());
            }}
            offset = offset.saturating_add(4096);
        }}
    }}

    #[inline]
    pub fn on_search_start() {{
        touch_pad();
        INIT.call_once(|| {{
            unsafe {{
                let _ = atexit(on_exit);
            }}
        }});
        sample();
    }}

    #[inline]
    pub fn sample() {{
        if let Ok(text) = std::fs::read_to_string("/proc/self/status") {{
            for line in text.lines() {{
                if let Some(rest) = line.strip_prefix("RssAnon:") {{
                    if let Some(num) = rest.split_whitespace().next() {{
                        if let Ok(kb) = num.parse::<u64>() {{
                            PEAK_RSS_ANON_KB.fetch_max(kb, Ordering::Relaxed);
                        }}
                    }}
                }} else if let Some(rest) = line.strip_prefix("VmHWM:") {{
                    if let Some(num) = rest.split_whitespace().next() {{
                        if let Ok(kb) = num.parse::<u64>() {{
                            PEAK_VM_HWM_KB.fetch_max(kb, Ordering::Relaxed);
                        }}
                    }}
                }}
            }}
        }}
    }}

    #[inline]
    pub fn note_basis_walk(evidence: u64) {{
        BASIS_WALKS.fetch_add(1, Ordering::Relaxed);
        BASIS_EVIDENCE.fetch_max(evidence, Ordering::Relaxed);
    }}

    #[inline]
    pub fn note_basis_claim() {{
        BASIS_CLAIMS.fetch_add(1, Ordering::Relaxed);
    }}

    #[inline]
    pub fn note_basis_probe() {{
        BASIS_PROBES.fetch_add(1, Ordering::Relaxed);
    }}

    #[inline]
    pub fn note_basis_candidate(spent: u64) {{
        BASIS_CANDIDATES.fetch_add(1, Ordering::Relaxed);
        BASIS_SPENT.fetch_add(spent, Ordering::Relaxed);
    }}
}}
"""


def atomic_write_text(path: pathlib.Path, content: str) -> str:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def instrument_tree(root: pathlib.Path, mode: str, pad_bytes: int) -> None:
    if pad_bytes < 1:
        pad_bytes = 1

    lib_path = root / "src" / "lib.rs"
    lib_text = lib_path.read_text(encoding="utf-8")
    mod_code = TELEMETRY_MOD_TEMPLATE.format(pad_bytes=pad_bytes)
    lib_text = replace_once(
        lib_text,
        "pub mod bias;\n",
        "pub mod bias;\n" + mod_code + "\n",
        "src/lib.rs:gate_telemetry",
    )
    atomic_write_text(lib_path, lib_text)

    bias_path = root / "src" / "bias.rs"
    bias_text = bias_path.read_text(encoding="utf-8")
    bias_text = replace_once(
        bias_text,
        "        self.centres.push(d);\n        self.visits.push(0);\n        self.centres.len() - 1\n",
        "        self.centres.push(d);\n        self.visits.push(0);\n        if self.centres.len().is_power_of_two() {\n            crate::gate_telemetry::sample();\n        }\n        self.centres.len() - 1\n",
        "src/bias.rs:BasinIndex::push",
    )
    atomic_write_text(bias_path, bias_text)

    ch_path = root / "src" / "methods" / "cluster_hopping.rs"
    ch_text = ch_path.read_text(encoding="utf-8")
    ch_text = replace_once(
        ch_text,
        "{\n    let _caller_stop = ledger.bind_stop();\n",
        "{\n    crate::gate_telemetry::on_search_start();\n    let _caller_stop = ledger.bind_stop();\n",
        "src/methods/cluster_hopping.rs:run_full_start",
    )
    ch_text = replace_once(
        ch_text,
        "    let final_radius = bias.merge_radius();\n    if let Some(slot) = carried {\n",
        "    crate::gate_telemetry::sample();\n    let final_radius = bias.merge_radius();\n    if let Some(slot) = carried {\n",
        "src/methods/cluster_hopping.rs:run_full_end",
    )

    if mode == "B":
        ch_text = replace_once(
            ch_text,
            "    let basis_evidence = (0..4).fold(1usize, |count, chosen| {\n        count.saturating_mul(n.saturating_sub(chosen)) / (chosen + 1)\n    });\n",
            "    let basis_evidence = (0..4).fold(1usize, |count, chosen| {\n        count.saturating_mul(n.saturating_sub(chosen)) / (chosen + 1)\n    });\n    crate::gate_telemetry::note_basis_walk(basis_evidence as u64);\n",
            "src/methods/cluster_hopping.rs:basis_evidence",
        )
        ch_text = replace_once(
            ch_text,
            "fn claim_observed_basis_probe(\n    cfg: &Config,\n    ledger: &Ledger,\n    evidence: usize,\n    basin: usize,\n    seen: &mut std::collections::HashSet<usize>,\n) -> bool {\n    !cfg.move_library.is_rigid_body()\n        && cfg.active_region.is_none()\n        && cfg.frozen.is_none()\n        && ledger.remaining() >= evidence\n        && seen.insert(basin)\n}\n",
            "fn claim_observed_basis_probe(\n    cfg: &Config,\n    ledger: &Ledger,\n    evidence: usize,\n    basin: usize,\n    seen: &mut std::collections::HashSet<usize>,\n) -> bool {\n    crate::gate_telemetry::note_basis_claim();\n    let claimed = !cfg.move_library.is_rigid_body()\n        && cfg.active_region.is_none()\n        && cfg.frozen.is_none()\n        && ledger.remaining() >= evidence\n        && seen.insert(basin);\n    if claimed {\n        crate::gate_telemetry::note_basis_probe();\n    }\n    claimed\n}\n",
            "src/methods/cluster_hopping.rs:claim_observed_basis_probe",
        )
        ch_text = replace_once(
            ch_text,
            "                    let side_spent = side_ledger.spent();\n                    ledger.budget = ledger.budget.saturating_sub(side_spent);\n                    observed_basis_spent = observed_basis_spent.saturating_add(side_spent);\n",
            "                    let side_spent = side_ledger.spent();\n                    crate::gate_telemetry::note_basis_candidate(side_spent as u64);\n                    crate::gate_telemetry::sample();\n                    ledger.budget = ledger.budget.saturating_sub(side_spent);\n                    observed_basis_spent = observed_basis_spent.saturating_add(side_spent);\n",
            "src/methods/cluster_hopping.rs:side_spent",
        )

        lat_path = root / "src" / "lattice.rs"
        lat_text = lat_path.read_text(encoding="utf-8")
        lat_text = replace_once(
            lat_text,
            "    let radii: Vec<f64> = sites\n        .iter()\n        .map(|&site| distance2(&lattice_point(site, basis), &centre))\n        .collect();\n    let mut alive = occupied;\n",
            "    let radii: Vec<f64> = sites\n        .iter()\n        .map(|&site| distance2(&lattice_point(site, basis), &centre))\n        .collect();\n    crate::gate_telemetry::sample();\n    let mut alive = occupied;\n",
            "src/lattice.rs:candidate_from_basis",
        )
        atomic_write_text(lat_path, lat_text)

    atomic_write_text(ch_path, ch_text)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: instrument_arms.py <worktree_root> <A|B> <pad_bytes>", file=sys.stderr)
        return 2
    root = pathlib.Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    if mode not in ("A", "B"):
        raise ValueError(f"unexpected mode: {mode}")
    pad_bytes = int(sys.argv[3])
    instrument_tree(root, mode, pad_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
