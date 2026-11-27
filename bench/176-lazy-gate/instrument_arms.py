#!/usr/bin/env python3
"""Insert in-process telemetry into a clean arm worktree.

The worktree is restored by the build wrapper after the binaries exist.
Nothing in this file is a search change: counters and hashes are reads.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

TELEMETRY_MOD = """
pub mod gate_telemetry {
    use std::io::Write;
    use std::sync::Once;
    use std::sync::atomic::{AtomicU64, Ordering};

    const PAD_BYTES: usize = __PAD_BYTES__;
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

    unsafe extern "C" {
        fn atexit(cb: extern "C" fn()) -> i32;
    }

    extern "C" fn on_exit() {
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
            "gate_telemetry rss_anon_kb={rss} vm_hwm_kb={hwm} basis_walks={walks} basis_evidence={evidence} basis_claims={claims} basis_probes={probes} basis_candidates={candidates} basis_spent={spent} pad_bytes={pad}",
            pad = PAD_BYTES,
        );
        let _ = out.flush();
    }

    #[inline]
    fn touch_pad() {
        let ptr = GATE_PAD.as_ptr();
        let mut offset = 0usize;
        while offset < PAD_BYTES {
            unsafe {
                std::hint::black_box(ptr.add(offset).read_volatile());
            }
            offset = offset.saturating_add(4096);
        }
    }

    #[inline]
    pub fn on_search_start() {
        touch_pad();
        INIT.call_once(|| {
            unsafe {
                let _ = atexit(on_exit);
            }
        });
        sample();
    }

    #[inline]
    pub fn sample() {
        if let Ok(text) = std::fs::read_to_string("/proc/self/status") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("RssAnon:") {
                    if let Some(num) = rest.split_whitespace().next() {
                        if let Ok(kb) = num.parse::<u64>() {
                            PEAK_RSS_ANON_KB.fetch_max(kb, Ordering::Relaxed);
                        }
                    }
                } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(num) = rest.split_whitespace().next() {
                        if let Ok(kb) = num.parse::<u64>() {
                            PEAK_VM_HWM_KB.fetch_max(kb, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }

    #[inline]
    pub fn note_basis_walk(evidence: u64) {
        BASIS_WALKS.fetch_add(1, Ordering::Relaxed);
        BASIS_EVIDENCE.fetch_max(evidence, Ordering::Relaxed);
    }

    #[inline]
    pub fn note_basis_claim() {
        BASIS_CLAIMS.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn note_basis_probe() {
        BASIS_PROBES.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn note_basis_candidate(spent: u64) {
        BASIS_CANDIDATES.fetch_add(1, Ordering::Relaxed);
        BASIS_SPENT.fetch_add(spent, Ordering::Relaxed);
    }

    pub fn emit_final(
        best_state: Option<&[f64]>,
        best_energy: f64,
        final_state: Option<&[f64]>,
        final_energy: f64,
    ) {
        fn hex_bits(values: Option<&[f64]>) -> String {
            let mut text = String::new();
            if let Some(values) = values {
                for value in values {
                    text.push_str(&format!("{:016x}", value.to_bits()));
                }
            }
            text
        }
        let mut out = std::io::stdout().lock();
        let _ = writeln!(
            out,
            "final_record energy_bits={:016x} struct_bits={} final_energy_bits={:016x} final_struct_bits={}",
            best_energy.to_bits(),
            hex_bits(best_state),
            final_energy.to_bits(),
            hex_bits(final_state),
        );
        let _ = out.flush();
    }
}
"""

CLAIM_LAZY_OLD = """fn claim_observed_basis_probe(
    cfg: &Config,
    ledger: &Ledger,
    evidence: usize,
    basin: impl FnOnce() -> usize,
    seen: &mut std::collections::HashSet<usize>,
) -> bool {
    !cfg.move_library.is_rigid_body()
        && cfg.active_region.is_none()
        && cfg.frozen.is_none()
        && ledger.remaining() >= evidence
        && seen.insert(basin())
}
"""

CLAIM_LAZY_NEW = """fn claim_observed_basis_probe(
    cfg: &Config,
    ledger: &Ledger,
    evidence: usize,
    basin: impl FnOnce() -> usize,
    seen: &mut std::collections::HashSet<usize>,
) -> bool {
    crate::gate_telemetry::note_basis_claim();
    let claimed = !cfg.move_library.is_rigid_body()
        && cfg.active_region.is_none()
        && cfg.frozen.is_none()
        && ledger.remaining() >= evidence
        && seen.insert(basin());
    if claimed {
        crate::gate_telemetry::note_basis_probe();
    }
    claimed
}
"""

CLAIM_DIRECT_OLD = """fn claim_observed_basis_probe(
    cfg: &Config,
    ledger: &Ledger,
    evidence: usize,
    basin: usize,
    seen: &mut std::collections::HashSet<usize>,
) -> bool {
    !cfg.move_library.is_rigid_body()
        && cfg.active_region.is_none()
        && cfg.frozen.is_none()
        && ledger.remaining() >= evidence
        && seen.insert(basin)
}
"""

CLAIM_DIRECT_NEW = """fn claim_observed_basis_probe(
    cfg: &Config,
    ledger: &Ledger,
    evidence: usize,
    basin: usize,
    seen: &mut std::collections::HashSet<usize>,
) -> bool {
    crate::gate_telemetry::note_basis_claim();
    let claimed = !cfg.move_library.is_rigid_body()
        && cfg.active_region.is_none()
        && cfg.frozen.is_none()
        && ledger.remaining() >= evidence
        && seen.insert(basin);
    if claimed {
        crate::gate_telemetry::note_basis_probe();
    }
    claimed
}
"""

EVIDENCE_OLD = """    let basis_evidence = (0..4).fold(1usize, |count, chosen| {
        count.saturating_mul(n.saturating_sub(chosen)) / (chosen + 1)
    });
"""

EVIDENCE_NEW = """    let basis_evidence = (0..4).fold(1usize, |count, chosen| {
        count.saturating_mul(n.saturating_sub(chosen)) / (chosen + 1)
    });
    crate::gate_telemetry::note_basis_walk(basis_evidence as u64);
"""

RADII_OLD = """    let radii: Vec<f64> = sites
        .iter()
        .map(|&site| distance2(&lattice_point(site, basis), &centre))
        .collect();
    let mut alive = occupied;
"""

RADII_NEW = """    let radii: Vec<f64> = sites
        .iter()
        .map(|&site| distance2(&lattice_point(site, basis), &centre))
        .collect();
    crate::gate_telemetry::sample();
    let mut alive = occupied;
"""

LAZY_CLOUD_OLD = """    if sites.len() < n {
        return None;
    }
    let mut alive = occupied;
"""

LAZY_CLOUD_NEW = """    if sites.len() < n {
        return None;
    }
    crate::gate_telemetry::sample();
    let mut alive = occupied;
"""

GMIN_OLD = (
    "    let (outcome, wall, pot_s) = search("
    "&cfg, spec.n, &start, seed, budget, spec.reference);\n"
)
GMIN_NEW = GMIN_OLD + """    let best_coords = outcome.best_state.as_ref().map(|state| state.to_vec());
    let final_coords = outcome.final_state.as_ref().map(|state| state.to_vec());
    anneal_core::gate_telemetry::emit_final(
        best_coords.as_deref(),
        outcome.best,
        final_coords.as_deref(),
        outcome.final_energy,
    );
"""

MORSE_OLD = (
    "    let (outcome, wall, pot_s) = search(&cfg, &spec, &start, seed, budget);\n"
)
MORSE_NEW = MORSE_OLD + """    let best_coords = outcome.best_state.as_ref().map(|state| state.to_vec());
    let final_coords = outcome.final_state.as_ref().map(|state| state.to_vec());
    anneal_core::gate_telemetry::emit_final(
        best_coords.as_deref(),
        outcome.best,
        final_coords.as_deref(),
        outcome.final_energy,
    );
"""

SEED_OLD = "    let seen = objective.seen.into_inner().unwrap();\n"
SEED_NEW = SEED_OLD + """    let best_coords = seen.at.to_vec();
    let final_coords = out.final_state.as_ref().map(|state| state.to_vec());
    anneal_core::gate_telemetry::emit_final(
        Some(best_coords.as_slice()),
        seen.lowest,
        final_coords.as_deref(),
        out.final_energy,
    );
"""


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    if "\r" in content:
        raise RuntimeError(f"refusing CR in {path.name}")
    if not content.endswith("\n"):
        content += "\n"
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != hashlib.sha256(content.encode("utf-8")).hexdigest():
        raise RuntimeError(f"read-back mismatch for {path.name}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def insert_after_exact_line(text: str, needle: str, added: list[str], label: str) -> str:
    lines = text.split("\n")
    hits = [index for index, line in enumerate(lines) if line.strip() == needle]
    if len(hits) != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {len(hits)}")
    index = hits[0]
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip(" "))]
    block = [indent + line for line in added]
    lines[index + 1 : index + 1] = block
    return "\n".join(lines)


def instrument_tree(root: pathlib.Path, mode: str, pad_bytes: int) -> None:
    if pad_bytes < 1:
        pad_bytes = 1
    if mode not in ("A", "B"):
        raise ValueError(f"unexpected mode: {mode}")

    lib_path = root / "src" / "lib.rs"
    lib_text = lib_path.read_text(encoding="utf-8")
    mod_code = TELEMETRY_MOD.replace("__PAD_BYTES__", str(pad_bytes))
    lib_text = replace_once(
        lib_text,
        "pub mod bias;\n",
        "pub mod bias;\n" + mod_code + "\n",
        "src/lib.rs",
    )
    atomic_write_text(lib_path, lib_text)

    bias_path = root / "src" / "bias.rs"
    bias_text = bias_path.read_text(encoding="utf-8")
    bias_text = replace_once(
        bias_text,
        "        self.centres.push(d);\n        self.visits.push(0);\n        self.centres.len() - 1\n",
        "        self.centres.push(d);\n        self.visits.push(0);\n        if self.centres.len().is_power_of_two() {\n            crate::gate_telemetry::sample();\n        }\n        self.centres.len() - 1\n",
        "src/bias.rs",
    )
    atomic_write_text(bias_path, bias_text)

    ch_path = root / "src" / "methods" / "cluster_hopping.rs"
    ch_text = ch_path.read_text(encoding="utf-8")
    ch_text = replace_once(
        ch_text,
        "{\n    let _caller_stop = ledger.bind_stop();\n",
        "{\n    crate::gate_telemetry::on_search_start();\n    let _caller_stop = ledger.bind_stop();\n",
        "run_full start",
    )
    ch_text = replace_once(
        ch_text,
        "    let final_radius = bias.merge_radius();\n    if let Some(slot) = carried {\n",
        "    crate::gate_telemetry::sample();\n    let final_radius = bias.merge_radius();\n    if let Some(slot) = carried {\n",
        "run_full end",
    )
    if mode == "B":
        if CLAIM_LAZY_OLD in ch_text:
            ch_text = replace_once(ch_text, CLAIM_LAZY_OLD, CLAIM_LAZY_NEW, "claim lazy")
        elif CLAIM_DIRECT_OLD in ch_text:
            ch_text = replace_once(ch_text, CLAIM_DIRECT_OLD, CLAIM_DIRECT_NEW, "claim direct")
        else:
            raise RuntimeError("claim_observed_basis_probe anchor missing")
        ch_text = replace_once(ch_text, EVIDENCE_OLD, EVIDENCE_NEW, "basis evidence")
        ch_text = insert_after_exact_line(
            ch_text,
            "let side_spent = side_ledger.spent();",
            [
                "crate::gate_telemetry::note_basis_candidate(side_spent as u64);",
                "crate::gate_telemetry::sample();",
            ],
            "side spent",
        )
        lat_path = root / "src" / "lattice.rs"
        lat_text = lat_path.read_text(encoding="utf-8")
        if RADII_OLD in lat_text:
            lat_text = replace_once(lat_text, RADII_OLD, RADII_NEW, "lattice radii")
        elif LAZY_CLOUD_OLD in lat_text:
            lat_text = replace_once(lat_text, LAZY_CLOUD_OLD, LAZY_CLOUD_NEW, "lattice cloud")
        else:
            raise RuntimeError("lattice anchor missing")
        atomic_write_text(lat_path, lat_text)
    atomic_write_text(ch_path, ch_text)

    for name, old, new in (
        ("h2h_gmin_proof.rs", GMIN_OLD, GMIN_NEW),
        ("h2h_morse.rs", MORSE_OLD, MORSE_NEW),
        ("h2h_seed.rs", SEED_OLD, SEED_NEW),
    ):
        path = root / "examples" / name
        text = path.read_text(encoding="utf-8")
        text = replace_once(text, old, new, name)
        atomic_write_text(path, text)


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: instrument_arms.py <worktree> <A|B> <pad_bytes>",
            file=sys.stderr,
        )
        return 2
    root = pathlib.Path(sys.argv[1]).resolve()
    instrument_tree(root, sys.argv[2], int(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
