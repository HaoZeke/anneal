# Copyright 2023--present rgpot developers
"""The shared library's rgpot_* export count stays at the measured pin.

49 for a normal link of librgpot_core.a, 95 when librgpot_bundle.a is
linked with --whole-archive. The extra 46 are the public C ABI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_rgpot_exports as exports


def _baseline(name: str) -> set[str]:
    return exports.read_baseline(ROOT / "scripts" / name)


def test_pinned_counts():
    assert exports.CORE_ONLY_EXPORTS == 49
    assert exports.BUNDLE_EXPORTS == 95
    assert exports.BUNDLE_EXPORTS - exports.CORE_ONLY_EXPORTS == 46


def test_baselines_are_the_public_archive_set():
    core = _baseline("abi_core_baseline.txt")
    fast = _baseline("abi_baseline_fast.txt")
    assert len(core) == 42
    assert len(fast) == 4
    assert core.isdisjoint(fast)
    assert len(core | fast) == 46


def test_whole_archive_exports_the_public_set():
    wanted = {f"rgpot_abi_{index}" for index in range(46)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exports.evaluate(umbrella | wanted, wanted, wanted, whole=True)


def test_core_only_exports_none_of_the_archive():
    wanted = {f"rgpot_abi_{index}" for index in range(42)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exports.evaluate(umbrella, wanted, wanted, whole=False)


def test_core_only_rejects_an_archive_symbol():
    wanted = {f"rgpot_abi_{index}" for index in range(42)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    umbrella.add("rgpot_abi_0")
    with pytest.raises(SystemExit, match="exported rust archive symbols"):
        exports.evaluate(umbrella, wanted, wanted, whole=False)


def test_whole_archive_rejects_a_dropped_name():
    wanted = {f"rgpot_abi_{index}" for index in range(46)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exported = (umbrella | wanted) - {"rgpot_abi_0"}
    with pytest.raises(SystemExit, match="dropped public"):
        exports.evaluate(exported, wanted, wanted, whole=True)


def test_count_mismatch_names_the_pin():
    wanted = {f"rgpot_abi_{index}" for index in range(46)}
    exported = {f"rgpot_cxx_{index}" for index in range(exports.BUNDLE_EXPORTS)}
    exported.update(wanted)
    with pytest.raises(SystemExit, match="expected 95"):
        exports.evaluate(exported, wanted, wanted, whole=True)


def test_nm_listing_keeps_defined_rgpot_names():
    text = "\n".join(
        [
            "0000000000000000 T rgpot_version",
            "                 U rgpot_potential_new",
            "0000000000000000 T not_rgpot",
            "0000000000000000 R rgpot_abi_stamp",
        ]
    )
    assert exports.rgpot_names(text) == {"rgpot_version", "rgpot_abi_stamp"}
