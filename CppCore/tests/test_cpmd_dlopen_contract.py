#!/usr/bin/env python3
"""Build contract checks for the CPMD pure-consumer boundary."""

from __future__ import annotations

import sys
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def has_symbol(text: str, symbol: str) -> bool:
    return re.search(rf"\b{re.escape(symbol)}\s*\(", text) is not None


def has_feature_entry(text: str, symbol: str) -> bool:
    return f'"abi.{symbol}"' in text


def has_literal_feature_entry(text: str, feature_id: str) -> bool:
    return f'"{feature_id}"' in text


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    cpmd_dir = root / "CppCore" / "rgpot" / "CPMDPot"
    meson = (cpmd_dir / "meson.build").read_text(encoding="utf-8")
    frontend = (cpmd_dir / "CPMDPot.cc").read_text(encoding="utf-8")
    header = (cpmd_dir / "cpmd_c_abi.h").read_text(encoding="utf-8")
    stub = (cpmd_dir / "cpmd_c_abi_stub.c").read_text(encoding="utf-8")
    feature_table = (cpmd_dir / "cpmd_feature_table.inc").read_text(
        encoding="utf-8"
    )
    fake = (root / "CppCore" / "tests" / "cpmdc_fake_engine.cc").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "RGPOT_CPMD_STATIC_EMBED",
        "cpmd_static_embed_dep",
        "cpmd_embed_static",
        "cpmd_extra_link",
    ]
    for token in forbidden:
        require(token not in meson, f"meson.build leaks static embed token {token}")
        require(token not in frontend, f"CPMDPot.cc leaks static embed token {token}")
    for token in ["with_cpmd", "cpmd_root"]:
        require(token not in meson, f"meson.build leaks static embed token {token}")

    require(
        "cpmdpot_dep = declare_dependency(" in meson,
        "CPMD frontend dependency is missing",
    )
    require(
        "dependencies: [cpmd_dl_dep, ptlrpc_dep]" in meson,
        "CPMD frontend dependency should expose dl plus the Cap'n Proto schema dependency",
    )
    require(
        "libcpmdc" in frontend,
        "CPMDPot.cc must dlopen the split cpmdc engine (libcpmdc)",
    )
    require(
        '#include "cpmd_feature_table.inc"' in stub,
        "cpmd_c_abi_stub.c must use the shared CPMD feature table",
    )
    require(
        '#include "cpmd_feature_table.inc"' in fake,
        "cpmdc_fake_engine.cc must use the shared CPMD feature table",
    )
    feature_symbols = [
        "cpmdc_feature_count",
        "cpmdc_feature_table",
        "cpmdc_feature_find",
    ]
    for symbol in feature_symbols:
        require(
            f'"{symbol}"' in frontend,
            f"CPMDPot.cc does not load feature discovery symbol {symbol}",
        )
    symbols = [
        "cpmdc_set_params",
        "cpmdc_energy_gradient",
        "cpmdc_energy",
        "cpmdc_energy_forces",
        "cpmdc_session_create",
        "cpmdc_session_set_params",
        "cpmdc_session_destroy",
        "cpmdc_session_select_orbitals",
        "cpmdc_session_copy_orbitals",
        "cpmdc_session_energy_gradient",
        "cpmdc_session_energy",
        "cpmdc_session_energy_forces",
        "cpmdc_session_calculate_forces",
        "cpmdc_session_calculate_result",
        "cpmdc_calculate_result",
        "cpmdc_potential_result_size_for_force_input",
        "cpmdc_version",
        "cpmdc_available",
        "cpmdc_finalize",
        "cpmdc_feature_count",
        "cpmdc_feature_table",
        "cpmdc_feature_find",
    ]
    for symbol in symbols:
        require(has_symbol(header, symbol), f"cpmd_c_abi.h missing {symbol}")
        require(has_symbol(stub, symbol), f"cpmd_c_abi_stub.c missing {symbol}")
        require(has_symbol(fake, symbol), f"cpmdc_fake_engine.cc missing {symbol}")
        require(
            has_feature_entry(feature_table, symbol),
            f"cpmd_feature_table.inc missing abi.{symbol}",
        )
    non_abi_features = [
        "section.system",
        "section.raw",
        "params.functional",
        "params.inputSections",
        "catalog.section.MOLSTATES",
        "catalog.cpmd.MOLECULAR_DYNAMICS_CP",
        "catalog.dft.FUNCTIONAL_PBE0",
    ]
    for feature_id in non_abi_features:
        require(
            has_literal_feature_entry(feature_table, feature_id),
            f"cpmd_feature_table.inc missing {feature_id}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
