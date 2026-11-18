# Copyright (c) anneal developers
"""Tests verifying that removed structure-prior keys error clearly in Python and CLI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import anneal

ROOT = Path(__file__).resolve().parents[1]

REMOVED_KEYS = (
    "prior_reseed",
    "prior_growth",
    "prior_core",
    "prior_handover",
    "prior_motif",
    "reseed_hops",
)


def test_python_config_validate_json_and_toml_reject_removed_keys():
    anneal.Config.validate_json('{"prior_soap": false, "continuous_symmetry": {"group": "measure", "interval": 10}}')
    anneal.Config.validate_toml('prior_soap = false\n[continuous_symmetry]\ngroup = "measure"\ninterval = 10\n')

    for key in REMOVED_KEYS:
        if key == "reseed_hops":
            payload_json = json.dumps(
                {"continuous_symmetry": {"group": "measure", "interval": 10, "reseed_hops": 5}}
            )
            payload_toml = '[continuous_symmetry]\ngroup = "measure"\ninterval = 10\nreseed_hops = 5\n'
        else:
            payload_json = json.dumps({key: False})
            payload_toml = f"{key} = false\n"

        with pytest.raises(ValueError, match="was removed"):
            anneal.Config.validate_json(payload_json)
        with pytest.raises(ValueError, match="was removed"):
            anneal.Config.validate_toml(payload_toml)


def test_python_config_and_cluster_search_do_not_expose_removed_priors():
    cfg = anneal.Config.recommended(38)
    for key in REMOVED_KEYS:
        assert not hasattr(cfg, key)
        with pytest.raises(AttributeError):
            getattr(cfg, key)

    def obj(x):
        return float(np.dot(x, x))

    def grad(x):
        return 2.0 * np.asarray(x, dtype=np.float64)

    for key in REMOVED_KEYS:
        with pytest.raises(TypeError):
            anneal.cluster_search(obj, grad, 3, 10, **{key: False})

    for dead_soap_fn in ("fivefold_plateau", "extended_scale", "motif_fractions"):
        assert not hasattr(anneal, dead_soap_fn)


def test_example_drivers_reject_removed_prior_env_switches(tmp_path: Path):
    subprocess.run(
        [
            "cargo",
            "build",
            "--features",
            "rgpot-ex",
            "--example",
            "h2h_morse",
            "--example",
            "h2h_metal",
            "--example",
            "h2h_covalent",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT,
            text=True,
        )
    )
    examples_dir = Path(meta["target_directory"]) / "debug" / "examples"

    for binary, args in (
        (examples_dir / "h2h_morse", ["run", "m38r6", "2000"]),
        (examples_dir / "h2h_metal", ["coords", "zn38", "2000", str(tmp_path / "coords.txt")]),
        (examples_dir / "h2h_covalent", ["coords", "sw10", "2000", str(tmp_path / "cov.txt")]),
    ):
        for env_override in ({"H2H_PRIORS": "reseed"}, {"H2H_HANDOVER": "0"}):
            env = dict(os.environ)
            env.pop("H2H_PRIORS", None)
            env.pop("H2H_HANDOVER", None)
            env.update(env_override)
            proc = subprocess.run(
                [str(binary), *args],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            assert proc.returncode != 0
            assert "was removed" in proc.stderr
