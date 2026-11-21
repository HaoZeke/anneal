#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Add the registered path counter to one clean benchmark arm."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

COUNTER = """
static STALL_NEAREST_INDEX_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

thread_local! {
    static COUNT_STALL_NEAREST_INDEX: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Number of nearest-index calls made by stall symmetrisation.
pub fn nearest_index_call_count() -> u64 {
    STALL_NEAREST_INDEX_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Count nearest-index calls made while one stall detector runs.
pub fn count_stall_nearest_index_calls<T>(run: impl FnOnce() -> T) -> T {
    let previous = COUNT_STALL_NEAREST_INDEX.with(|enabled| enabled.replace(true));
    let result = run();
    COUNT_STALL_NEAREST_INDEX.with(|enabled| enabled.set(previous));
    result
}

"""

INCREMENT = """    fn nearest_index(&self, image: [f64; 3]) -> (usize, f64) {
        COUNT_STALL_NEAREST_INDEX.with(|enabled| {
            if enabled.get() {
                STALL_NEAREST_INDEX_CALLS.fetch_add(
                    1,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        });
"""

STALL_OLD = """                let cands =
                    crate::symmetrise::detect_orders(x.view(), n, &orders, cfg.symmetry_tolerance);
"""

STALL_NEW = """                let cands = crate::symmetrise::count_stall_nearest_index_calls(|| {
                    crate::symmetrise::detect_orders(
                        x.view(),
                        n,
                        &orders,
                        cfg.symmetry_tolerance,
                    )
                });
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected one {label} anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    symmetrise = repo / "src" / "symmetrise.rs"
    hopping = repo / "src" / "methods" / "cluster_hopping.rs"

    dirty = git(repo, "status", "--short", "--", str(symmetrise), str(hopping))
    if dirty:
        raise SystemExit("refusing an arm with modified counter targets")

    text = symmetrise.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "/// Centred coordinates in separate arrays, so a nearest-image scan reads\n",
        COUNTER
        + "/// Centred coordinates in separate arrays, so a nearest-image scan reads\n",
        "counter insertion",
    )
    text = replace_once(
        text,
        "    fn nearest_index(&self, image: [f64; 3]) -> (usize, f64) {\n",
        INCREMENT,
        "nearest-index function",
    )
    symmetrise.write_text(text, encoding="utf-8", newline="\n")

    text = hopping.read_text(encoding="utf-8")
    text = replace_once(text, STALL_OLD, STALL_NEW, "stall detector")
    hopping.write_text(text, encoding="utf-8", newline="\n")

    print(f"commit={git(repo, 'rev-parse', 'HEAD')}")
    print(f"tree={git(repo, 'rev-parse', 'HEAD^{tree}')}")
    print(f"symmetrise_sha256={sha256(symmetrise)}")
    print(f"cluster_hopping_sha256={sha256(hopping)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
