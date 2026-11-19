"""The RPC end to end file runs in the rpctest task and skips only outside it."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tests" / "test_rpc_e2e_c_abi.py"
SCRIPT = ROOT / "scripts" / "run_rpctest.sh"

# A direct run outside the task prints this sentence. The task and
# `potctl ci rpc-integ` set RGPOT_RPC_E2E_REQUIRED so a missing binary fails.
OUTSIDE = "Outside the rpctest task this case is skipped."
ENGINE_VARS = (
    "RGPOT_POTSERV",
    "NWCHEMC_LIBRARY",
    "RGPOT_NWCHEMC_ENGINE",
    "RGPOT_NWCHEM_ENGINE",
    "CPMDC_LIBRARY",
    "RGPOT_CPMDC_ENGINE",
    "RGPOT_CPMD_ENGINE",
    "RGPOT_RPC_E2E_REQUIRED",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ENGINE_VARS:
        env.pop(name, None)
    return env


def test_rpctest_task_exports_potserv_and_fake_engines() -> None:
    data = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    tasks = data["feature"]["itests"]["tasks"]
    assert "rpctest" in tasks
    text = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "RGPOT_POTSERV",
        "NWCHEMC_LIBRARY",
        "RGPOT_NWCHEM_ENGINE",
        "CPMDC_LIBRARY",
        "RGPOT_CPMD_ENGINE",
        "RGPOT_RPC_E2E_REQUIRED",
    ):
        assert f"export {name}" in text
    assert "libnwchemc_fake_engine" in text
    assert "libcpmdc_fake_engine" in text
    assert "potctl" in text
    assert "rpc-integ" in text


def test_direct_pytest_skips_with_a_reason() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-rs", str(E2E)],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "2 skipped" in combined
    assert OUTSIDE in combined
    assert "RGPOT_POTSERV" in combined


def test_required_run_does_not_skip_when_potserv_is_absent(tmp_path: Path) -> None:
    env = _clean_env()
    env["RGPOT_RPC_E2E_REQUIRED"] = "1"
    env["RGPOT_POTSERV"] = str(tmp_path / "missing-potserv")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line", str(E2E)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "SKIPPED" not in combined
    assert "skipped in" not in combined
    assert "RGPOT_POTSERV" in combined


def test_running_the_file_does_not_raise_skipped() -> None:
    proc = subprocess.run(
        [sys.executable, str(E2E)],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined, combined
    assert OUTSIDE in combined
    assert proc.returncode != 0


def test_eon_page_matches_the_skip() -> None:
    text = (ROOT / "docs" / "orgmode" / "howto" / "eon-rgpot.org").read_text(
        encoding="utf-8"
    )
    assert "fails when one is missing" not in text
    assert "pixi run -e rpctest rpctest" in text
    assert "potctl ci rpc-integ" in text


def test_python_api_metatomic_flag_matches_the_binding() -> None:
    text = (ROOT / "docs" / "orgmode" / "reference" / "python_api.org").read_text(
        encoding="utf-8"
    )
    start = text.index("*** =rgpot.has_metatomic_dlopen=")
    window = text[start : text.index("*** =rgpot.has_xckernel=")]
    assert "compiled with =RGPOT_HAS_METATOMIC=" not in window
    assert "every" in window.lower()
    calls = text.index("*** =rgpot.evaluate_metatomic=")
    calls = text[calls : text.index("*** =rgpot.default_metatomic_engine_path=")]
    assert "has no MetatomicDlopen frontend" not in calls
    assert "has_metatomic_dlopen= is false" not in calls
    howto = (ROOT / "docs" / "orgmode" / "howto" / "pip-python.org").read_text(
        encoding="utf-8"
    )
    assert "has_metatomic_dlopen= is =False=" not in howto


def test_librgpot_soname_is_an_installed_target() -> None:
    meson = (ROOT / "meson.build").read_text(encoding="utf-8")
    start = meson.index("foreach _tgt_name")
    block = meson[start : meson.index("endforeach", start)]
    assert "'librgpot_soname'" in block
    python_meson = (ROOT / "python" / "meson.build").read_text(encoding="utf-8")
    assigned = False
    for line in python_meson.splitlines():
        stripped = line.strip()
        if stripped.startswith("librgpot_soname") and "custom_target" in stripped:
            assigned = True
    assert assigned
