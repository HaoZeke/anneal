# Copyright (c) anneal developers
"""Provenance and call-accounting checks for the GMIN head-to-head."""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from machine_provenance import MACHINE_ID_SALT  # noqa: E402

RUNNER = ROOT / "scripts" / "gmin_h2h" / "run_campaign.py"
SPEC = importlib.util.spec_from_file_location("gmin_h2h_campaign", RUNNER)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_machine_provenance_prefers_machine_id_and_records_cpu(tmp_path):
    machine_id = tmp_path / "machine-id"
    product_uuid = tmp_path / "product_uuid"
    cpuinfo = tmp_path / "cpuinfo"
    machine_id.write_text(" machine-a \n")
    product_uuid.write_text("fallback-b\n")
    cpuinfo.write_text("processor: 0\nmodel name: Example  CPU   123\n")

    digest, cpu = campaign.machine_provenance(machine_id, product_uuid, cpuinfo)

    expected = hashlib.sha256(MACHINE_ID_SALT + b"machine-a").hexdigest()
    assert digest == expected
    assert cpu == "Example CPU 123"


def test_machine_provenance_falls_back_to_dmi_uuid(tmp_path):
    missing = tmp_path / "missing"
    product_uuid = tmp_path / "product_uuid"
    cpuinfo = tmp_path / "cpuinfo"
    product_uuid.write_text("dmi-machine\n")
    cpuinfo.write_text("Hardware : Example Board\n")

    digest, cpu = campaign.machine_provenance(missing, product_uuid, cpuinfo)

    expected = hashlib.sha256(MACHINE_ID_SALT + b"dmi-machine").hexdigest()
    assert digest == expected
    assert cpu == "Example Board"


def test_summary_keeps_host_and_adds_physical_machine_provenance():
    rows = [
        {
            "n": 98,
            "arm": "anneal",
            "seed": 2000,
            "budget": 4_000_000,
            "hit": False,
            "evals": 4_000_000.0,
            "wall_s": 1.0,
            "best": -1.0,
        }
    ]
    summary = campaign.summarise(
        rows,
        (2000,),
        "https://example.invalid/gmin",
        "gmin-revision",
        "anneal-revision",
        "a" * 64,
        "Example CPU",
    )

    assert summary[0]["host"] == os.uname().nodename
    assert summary[0]["machine_id"] == "a" * 64
    assert summary[0]["cpu_model"] == "Example CPU"
    assert tuple(summary[0]) == tuple(campaign.SUMMARY_FIELDS)


def test_per_seed_rows_carry_machine_provenance_and_gmin_counters(tmp_path):
    path = tmp_path / "gmin.row"
    campaign.write_per_seed_row(
        path,
        {
            "system": "lj98",
            "code": "gmin",
            "seed": "2000",
            "budget": "4000000",
            "hit": "0",
            "evals_to_hit": "4000000",
            "charged": "4000000",
            "gradient_passes": "4000000",
            "energy_only_entries": "27",
            "wall_s": "1.0",
            "best": "-1.0",
            "pot_s": "0.9",
        },
        "a" * 64,
        "Example CPU, revision 2",
    )

    row = campaign.parse_gmin(path.read_text())
    assert row["gradient_passes"] == "4000000"
    assert row["energy_only_entries"] == "27"
    assert row["machine_id"] == "a" * 64
    assert row["cpu_model"] == "Example CPU, revision 2"


def test_gmin_cap_charges_gradient_passes_only(tmp_path):
    gfortran = shutil.which("gfortran")
    if gfortran is None:
        pytest.skip("gfortran is not on PATH")
    (tmp_path / "prec_stub.f90").write_text(
        "MODULE PREC\n"
        "  INTEGER, PARAMETER :: REAL64=SELECTED_REAL_KIND(15,307)\n"
        "END MODULE PREC\n"
    )
    (tmp_path / "driver.f90").write_text(
        "PROGRAM CHECK_CAP\n"
        "  USE H2H_CAP\n"
        "  LOGICAL :: REFUSED\n"
        "  REFUSED=H2H_REFUSE(.FALSE.)\n"
        "  IF (REFUSED) STOP 1\n"
        "  REFUSED=H2H_REFUSE(.TRUE.)\n"
        "  IF (REFUSED) STOP 2\n"
        "  REFUSED=H2H_REFUSE(.FALSE.)\n"
        "  IF (REFUSED) STOP 3\n"
        "  REFUSED=H2H_REFUSE(.TRUE.)\n"
        "  IF (REFUSED) STOP 4\n"
        "  REFUSED=H2H_REFUSE(.TRUE.)\n"
        "  IF (.NOT. REFUSED) STOP 5\n"
        "  CALL H2H_REPORT()\n"
        "END PROGRAM CHECK_CAP\n"
    )
    binary = tmp_path / "check-cap"
    subprocess.run(
        [
            gfortran,
            str(tmp_path / "prec_stub.f90"),
            str(ROOT / "bench" / "lj104-lattice" / "v2" / "h2h_cap.f90"),
            str(tmp_path / "driver.f90"),
            "-o",
            str(binary),
        ],
        check=True,
        cwd=tmp_path,
    )
    row_path = tmp_path / "row.txt"
    env = dict(os.environ, H2H_BUDGET="2", H2H_ROW=str(row_path))
    subprocess.run([str(binary)], check=True, cwd=tmp_path, env=env)
    row = campaign.parse_gmin(row_path.read_text())

    assert row["budget"] == "2"
    assert row["charged"] == "2"
    assert row["gradient_passes"] == "2"
    assert row["energy_only_entries"] == "2"
