#!/usr/bin/env python3
"""Add one in-process RssAnon sampler to a source tree.

The sampler is measurement only. It does not change a preset, a stop, a
ledger, or a move. The build restores the three source files after the
binary exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODULE = r'''
pub mod gate_memory {
    use std::io::Write;
    use std::sync::Once;
    use std::sync::atomic::{AtomicU64, Ordering};

    static INIT: Once = Once::new();
    static PEAK_RSS_ANON_KB: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> i32;
    }

    extern "C" fn report() {
        sample();
        let peak = PEAK_RSS_ANON_KB.load(Ordering::Relaxed);
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "gate_memory peak_rss_anon_kb={peak}");
        let _ = out.flush();
    }

    pub fn start() {
        INIT.call_once(|| unsafe {
            let _ = atexit(report);
        });
        sample();
    }

    pub fn sample() {
        if let Ok(text) = std::fs::read_to_string("/proc/self/status") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("RssAnon:")
                    && let Some(value) = rest.split_whitespace().next()
                    && let Ok(kb) = value.parse::<u64>()
                {
                    PEAK_RSS_ANON_KB.fetch_max(kb, Ordering::Relaxed);
                }
            }
        }
    }
}
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def instrument(root: Path) -> None:
    library = root / "src" / "lib.rs"
    text = library.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "pub mod bias;\n",
        "pub mod bias;\n" + MODULE + "\n",
        "memory module",
    )
    write(library, text)

    hopping = root / "src" / "methods" / "cluster_hopping.rs"
    text = hopping.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    let _caller_stop = ledger.bind_stop();\n",
        "    crate::gate_memory::start();\n"
        "    let _caller_stop = ledger.bind_stop();\n",
        "search start",
    )
    text = replace_once(
        text,
        "    let final_radius = bias.merge_radius();\n",
        "    crate::gate_memory::sample();\n"
        '    println!("gate_walks hops={hops}");\n'
        "    let final_radius = bias.merge_radius();\n",
        "search end",
    )
    write(hopping, text)

    bias = root / "src" / "bias.rs"
    text = bias.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "        self.visits.push(0);\n",
        "        self.visits.push(0);\n"
        "        if self.centres.len().is_power_of_two() {\n"
        "            crate::gate_memory::sample();\n"
        "        }\n",
        "basin growth",
    )
    write(bias, text)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: instrument.py SOURCE_TREE", file=sys.stderr)
        return 2
    instrument(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
