# Copyright (c) anneal developers
"""Shared benchmark machine-provenance helpers."""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from machine_provenance import (  # noqa: E402
    aggregate_machine_columns,
    csv_fields,
    require_csv_schema,
)

ACTIVE_ROW_WRITERS = (
    "scripts/ase_nonlj/run_one.py",
    "scripts/ase_nonlj/summarize.py",
    "scripts/ase_panel/run_one.py",
    "scripts/ase_panel/summarize.py",
    "scripts/covalent_h2h/gfn2_crest.py",
    "scripts/covalent_h2h/gfn2_floor_gate.py",
    "scripts/covalent_h2h/reference.py",
    "scripts/covalent_h2h/run_campaign.py",
    "scripts/gad_published/run_campaign.py",
    "scripts/generic_alloy/run_ablation.py",
    "scripts/gmin_h2h/run_campaign.py",
    "scripts/gmin_panel/run_losses2.py",
    "scripts/gmin_panel/run_panel.py",
    "scripts/growpal_h2h/run_campaign.py",
    "scripts/growpal_h2h_v2/run_campaign.py",
    "scripts/lj_bonobo_h2h/run_campaign.py",
    "scripts/metal_h2h/run_campaign.py",
    "scripts/metal_h2h/enter_gold.sh",
    "scripts/metal_h2h/run_darby.py",
    "scripts/metal_h2h/run_ptcu.py",
    "scripts/metal_h2h/run_sc.py",
    "scripts/metal_h2h/run_sc96_gmin.py",
    "scripts/metal_h2h/run_sc_n12.py",
    "scripts/metal_h2h/run_sc_rows.py",
    "scripts/metal_h2h/summarise_sc_chain.py",
    "scripts/metal_h2h/summarise_sc_pop.py",
    "scripts/metal_h2h/summarise_sc_portfolio.py",
    "scripts/metal_h2h/summarise_sc_restart.py",
    "scripts/morse_gmin/run_panel.py",
    "scripts/morse_h2h/run_campaign.py",
    "scripts/morse_panel/run_panel.py",
    "scripts/occat_h2h/run_campaign.py",
    "scripts/tip4p_h2h/run_budget.py",
    "scripts/tip4p_h2h/run_h2h.py",
    "scripts/tip4p_h2h/run_water_gmin.py",
)


def test_require_csv_schema_refuses_legacy_header(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("host\nbench\n")

    with pytest.raises(RuntimeError, match="legacy columns"):
        require_csv_schema(path, ("host", "machine_id", "cpu_model"))

    assert path.read_text() == "host\nbench\n"


def test_require_csv_schema_accepts_current_header(tmp_path):
    path = tmp_path / "rows.csv"
    fields = ("host", "machine_id", "cpu_model")
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerow(fields)

    require_csv_schema(path, fields)


def test_aggregate_machine_columns_preserves_legacy_groups():
    assert aggregate_machine_columns([{"host": "bench"}]) == {}


def test_aggregate_machine_columns_rejects_mixed_groups():
    with pytest.raises(RuntimeError, match="cannot mix"):
        aggregate_machine_columns(
            [
                {"host": "bench"},
                {"machine_id": "a" * 64, "cpu_model": "Example CPU"},
            ]
        )


def test_csv_fields_quotes_cpu_models_with_commas():
    assert csv_fields(("a" * 64, "Example CPU, revision 2")) == (
        "a" * 64 + ',"Example CPU, revision 2"'
    )


@pytest.mark.parametrize("relative", ACTIVE_ROW_WRITERS)
def test_active_row_writers_include_shared_machine_provenance(relative):
    source = (ROOT / relative).read_text()

    assert "machine_provenance" in source
    assert any(
        helper in source
        for helper in (
            "machine_columns",
            "machine_provenance(",
            "aggregate_machine_columns",
            "csv_fields",
        )
    )


def _load_module(name: str, relative: str):
    import importlib.util

    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGED_RUNNERS = (
    ("scripts/generic_alloy/enter_bench.sh", "run_ablation.py"),
    ("scripts/growpal_h2h/enter_bench.sh", "run_campaign.py"),
    ("scripts/growpal_h2h_v2/enter_bench.sh", "run_campaign.py"),
    ("scripts/lj_bonobo_h2h/enter_bench.sh", "run_campaign.py"),
    ("scripts/metal_h2h/enter_bench.sh", "run_campaign.py"),
    ("scripts/metal_h2h/enter_sc_rows.sh", "run_sc_rows.py"),
    ("scripts/gmin_panel/enter_bench.sh", "run_panel.py"),
    ("scripts/gmin_panel/enter_losses2.sh", "run_losses2.py"),
    ("scripts/gad_published/enter_bench.sh", "run_campaign.py"),
    ("scripts/morse_h2h/enter_bench.sh", "run_campaign.py"),
    ("scripts/gmin_h2h/enter_bench.sh", "run_campaign.py"),
    ("scripts/covalent_h2h/enter_bench.sh", "run_campaign.py"),
    ("scripts/tip4p_h2h/enter_h2h.sh", "run_h2h.py"),
    ("scripts/tip4p_h2h/enter_budget.sh", "run_budget.py"),
    ("scripts/tip4p_h2h/enter_water_gmin.sh", "run_water_gmin.py"),
    ("scripts/metal_h2h/enter_darby.sh", "run_darby.py"),
    ("scripts/metal_h2h/enter_ptcu.sh", "run_ptcu.py"),
    ("scripts/metal_h2h/enter_sc.sh", "run_sc.py"),
    ("scripts/metal_h2h/enter_sc_n12.sh", "run_sc_n12.py"),
    ("scripts/metal_h2h/enter_sc96_gmin.sh", "run_sc96_gmin.py"),
)


def _stage_from_enter_script(enter_rel: str, dest: Path) -> None:
    import os
    import shlex
    import shutil

    script_path = ROOT / enter_rel
    here = str(script_path.parent)
    root = str(ROOT)
    lines: list[str] = []
    carry = ""
    for raw in script_path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            carry += stripped[:-1] + " "
        else:
            lines.append(carry + stripped)
            carry = ""
    if carry:
        lines.append(carry)

    dest.mkdir(parents=True, exist_ok=True)
    staged_any = False
    for line in lines:
        if not line.startswith("cp "):
            continue
        tokens = shlex.split(line)
        args = [tok for tok in tokens[1:] if not tok.startswith("-")]
        if len(args) < 2:
            continue
        target = args[-1]
        if not any(marker in target for marker in ("$WORK", "$OUT")):
            continue
        for src_tok in args[:-1]:
            if "$HERE" not in src_tok and "$ROOT" not in src_tok:
                continue
            expanded = src_tok.replace("$HERE", here).replace("$ROOT", root)
            src_path = Path(expanded)
            if src_path.is_file():
                target_name = Path(target).name if target.endswith(".py") else src_path.name
                shutil.copy2(src_path, dest / target_name)
                staged_any = True
    assert staged_any, f"no staged files extracted from {enter_rel}"


@pytest.mark.parametrize(("enter_rel", "runner_name"), STAGED_RUNNERS)
def test_staged_enter_scripts_execute_runner_from_isolated_temp_dir(
    tmp_path, enter_rel, runner_name
):
    import os
    import subprocess

    stage_dir = tmp_path / "work" / "bin"
    _stage_from_enter_script(enter_rel, stage_dir)
    staged_runner = stage_dir / runner_name
    assert staged_runner.is_file()

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONSTARTUP")
    }
    env["PYTHONPATH"] = str(stage_dir)
    env["HOME"] = str(tmp_path)

    cli = subprocess.run(
        [sys.executable, str(staged_runner), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (cli.stdout or "") + (cli.stderr or "")
    assert "ModuleNotFoundError" not in combined
    assert "ImportError" not in combined
    assert cli.returncode in (0, 2)

    loader = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, pathlib; "
                f"p = pathlib.Path({str(staged_runner)!r}); "
                "spec = importlib.util.spec_from_file_location('staged_runner', p); "
                "mod = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(mod)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert loader.returncode == 0, loader.stderr


def test_gmin_panel_rejects_mixed_gmin_revisions_and_emits_machine_columns(tmp_path):
    panel = _load_module("gmin_panel_run_panel", "scripts/gmin_panel/run_panel.py")
    losses2 = _load_module("gmin_panel_run_losses2", "scripts/gmin_panel/run_losses2.py")

    assert "cpu" not in panel.ROW_FIELDS
    assert "machine_id" in panel.ROW_FIELDS
    assert "cpu_model" in panel.ROW_FIELDS

    def make_row(code: str, seed: int, code_revision: str, machine_id: str, cpu_model: str):
        return {
            "system": "lj38",
            "code": code,
            "driver": "GMIN" if code == "gmin" else "h2h_gmin_proof",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.500000",
            "best": "-173.928427",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "code_revision": code_revision,
            "harness_revision": "head-rev",
            "host": "bench-a" if code != "gmin" else "bench-b",
            "machine_id": machine_id,
            "cpu_model": cpu_model,
            "session": "20261121T000000Z",
            "utc": "2026-11-21T00:00:00Z",
        }

    mixed_rows = [
        make_row("anneal", seed, "head-rev", "a" * 64, "CPU A") for seed in panel.SEEDS
    ] + [
        make_row(
            "gmin",
            seed,
            "svn-100" if seed != panel.SEEDS[-1] else "svn-101",
            "b" * 64,
            "CPU B",
        )
        for seed in panel.SEEDS
    ]
    with pytest.raises(RuntimeError, match="mixed code_revision"):
        panel.panel_groups(mixed_rows, "head-rev")
    with pytest.raises(RuntimeError, match="mixed code_revision"):
        losses2.selected(mixed_rows, "lj38", "gmin", "head-rev", "base-rev")

    clean_rows = [
        make_row("anneal", seed, "head-rev", "a" * 64, "CPU A") for seed in panel.SEEDS
    ] + [
        make_row("gmin", seed, "svn-100", "b" * 64, "CPU B") for seed in panel.SEEDS
    ]
    rows_csv = tmp_path / "rows.csv"
    for row in clean_rows:
        panel.append_row(rows_csv, row)
    require_csv_schema(rows_csv, panel.ROW_FIELDS)

    bad_row = dict(clean_rows[0], machine_id="")
    with pytest.raises(ValueError, match="machine_id"):
        panel.append_row(rows_csv, bad_row)

    legacy_csv = tmp_path / "legacy.csv"
    legacy_fields = [col for col in panel.ROW_FIELDS if col not in ("machine_id", "cpu_model")] + ["cpu"]
    legacy_csv.write_text(",".join(legacy_fields) + "\n")
    with pytest.raises(RuntimeError, match="legacy columns"):
        panel.append_row(legacy_csv, clean_rows[0])

    summary = panel.summarise(clean_rows, "head-rev")
    assert len(summary) == 1
    assert summary[0]["machine_id"] == f"{'a' * 64};{'b' * 64}"
    assert summary[0]["cpu_model"] == "CPU A;CPU B"
    fake_root = tmp_path / "root"
    readme_path = fake_root / panel.README
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        f"{panel.README_MARKS[0]}\nold\n{panel.README_MARKS[1]}\n"
    )
    summary_csv = tmp_path / "out" / "summary.csv"
    panel.write_tables(fake_root, clean_rows, "head-rev", False, summary_csv)
    require_csv_schema(
        summary_csv,
        (*panel.SUMMARY_FIELDS, "machine_id", "cpu_model"),
    )

    losses_rows = [
        make_row("baseline", seed, "base-rev", "a" * 64, "CPU A") for seed in panel.SEEDS
    ] + [
        make_row("fix2", seed, "head-rev", "a" * 64, "CPU A") for seed in panel.SEEDS
    ] + [
        make_row("gmin", seed, "svn-100", "b" * 64, "CPU B") for seed in panel.SEEDS
    ]
    losses_summary = losses2.summarise(losses_rows, "head-rev", "base-rev")
    by_pair = {(row["left"], row["right"]): row for row in losses_summary}
    assert by_pair[("fix2", "gmin")]["machine_id"] == f"{'a' * 64};{'b' * 64}"
    assert by_pair[("fix2", "gmin")]["cpu_model"] == "CPU A;CPU B"
    losses_summary_path = tmp_path / "losses-summary.csv"
    losses2.write_summary(losses_summary_path, losses_summary)
    require_csv_schema(
        losses_summary_path,
        (*losses2.SUMMARY_FIELDS, "machine_id", "cpu_model"),
    )


def test_tip4p_budget_preserves_reused_gmin_provenance_and_aggregates_summary(tmp_path):
    budget_mod = _load_module("tip4p_run_budget", "scripts/tip4p_h2h/run_budget.py")

    gmin_csv = tmp_path / "gmin-rows.csv"
    gmin_fields = [
        "system",
        "code",
        "seed",
        "budget",
        "hit",
        "evals_to_hit",
        "charged",
        "best",
        "wall_s",
        "pot_s",
        "outside_s",
        "code_revision",
        "host",
        "machine_id",
        "cpu_model",
    ]
    target_energy = budget_mod.CAMBRIDGE[budget_mod.SIZE]
    with gmin_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gmin_fields, lineterminator="\n")
        writer.writeheader()
        for seed in budget_mod.SEEDS:
            writer.writerow(
                {
                    "system": f"tip4p{budget_mod.SIZE}",
                    "code": "gmin",
                    "seed": str(seed),
                    "budget": str(budget_mod.BUDGETS[0]),
                    "hit": "1",
                    "evals_to_hit": "25000",
                    "charged": "25000",
                    "best": str(target_energy),
                    "wall_s": "1.200000",
                    "pot_s": "1.000000",
                    "outside_s": "0.200000",
                    "code_revision": "svn-remote",
                    "host": "remote-host",
                    "machine_id": "b" * 64,
                    "cpu_model": "Remote CPU",
                }
            )

    reused = budget_mod.load_reuse(gmin_csv, "svn-remote")
    assert len(reused) == len(budget_mod.SEEDS)
    assert budget_mod.load_reuse(gmin_csv, "svn-other") == {}

    local_machine = {"machine_id": "a" * 64, "cpu_model": "Local CPU"}
    rows = []
    for seed in budget_mod.SEEDS:
        gmin_row = budget_mod.build_budget_row(
            "gmin",
            reused[(budget_mod.BUDGETS[0], seed)],
            n=budget_mod.SIZE,
            budget=budget_mod.BUDGETS[0],
            seed=seed,
            head="anneal-head-1",
            revision="svn-remote",
            host="local-host",
            machine=local_machine,
            reused=True,
        )
        assert gmin_row["host"] == "remote-host"
        assert gmin_row["machine_id"] == "b" * 64
        assert gmin_row["cpu_model"] == "Remote CPU"
        assert gmin_row["code_revision"] == "svn-remote"
        rows.append(gmin_row)

        anneal_row = budget_mod.build_budget_row(
            "anneal",
            {
                "hit": "1",
                "evals_to_hit": "15000",
                "charged": "15000",
                "best": str(target_energy),
                "wall_s": "0.600000",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
            },
            n=budget_mod.SIZE,
            budget=budget_mod.BUDGETS[0],
            seed=seed,
            head="anneal-head-1",
            revision="svn-remote",
            host="local-host",
            machine=local_machine,
            reused=False,
        )
        assert anneal_row["host"] == "local-host"
        assert anneal_row["machine_id"] == "a" * 64
        assert anneal_row["cpu_model"] == "Local CPU"
        rows.append(anneal_row)

    with pytest.raises(RuntimeError, match="without machine provenance"):
        budget_mod.build_budget_row(
            "gmin",
            {**reused[(budget_mod.BUDGETS[0], budget_mod.SEEDS[0])], "machine_id": ""},
            n=budget_mod.SIZE,
            budget=budget_mod.BUDGETS[0],
            seed=budget_mod.SEEDS[0],
            head="anneal-head-1",
            revision="svn-remote",
            host="local-host",
            machine=local_machine,
            reused=True,
        )

    summary = budget_mod.summarise_budgets(
        rows,
        (budget_mod.BUDGETS[0],),
        budget_mod.SIZE,
    )
    assert len(summary) == 1
    assert summary[0]["machine_id"] == f"{'a' * 64};{'b' * 64}"
    assert summary[0]["cpu_model"] == "Local CPU;Remote CPU"


def test_remaining_gmin_runners_emit_machine_id_and_cpu_model(tmp_path):
    gad = _load_module("gad_published_campaign", "scripts/gad_published/run_campaign.py")
    morse_gmin = _load_module("morse_gmin_panel", "scripts/morse_gmin/run_panel.py")
    morse_h2h = _load_module("morse_h2h_campaign", "scripts/morse_h2h/run_campaign.py")
    morse_panel = _load_module("morse_panel_runner", "scripts/morse_panel/run_panel.py")

    machine_a = {"machine_id": "a" * 64, "cpu_model": "CPU A"}
    machine_b = {"machine_id": "b" * 64, "cpu_model": "CPU B"}

    gad.self_test()
    gad_row_path = tmp_path / "gad-anneal.row"
    gad.write_saved_rerun(
        gad_row_path,
        "anneal",
        {
            "system": "lj38",
            "seed": "2000",
            "budget": str(gad.BUDGET),
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.5",
            "best": "-173.928427",
            "pot_s": "0.4",
        },
        code_revision="head-1",
        host="bench",
        machine=machine_a,
    )
    assert (
        gad.saved_rerun(
            gad_row_path,
            38,
            "anneal",
            code_revision="head-1",
            machine=machine_a,
        )
        is not None
    )
    assert (
        gad.saved_rerun(
            gad_row_path,
            38,
            "anneal",
            code_revision="head-2",
            machine=machine_a,
        )
        is None
    )
    gad_rec = gad.saved_rerun(
        gad_row_path, 38, "anneal", code_revision="head-1", machine=machine_a
    )
    gad_summary = gad.summarise([gad_rec], "head-1", "svn-1", "bench", machine_b)
    assert gad_summary[0]["machine_id"] == "a" * 64
    assert gad_summary[0]["cpu_model"] == "CPU A"
    gad_csv = tmp_path / "gad-summary.csv"
    gad.append_summary(gad_csv, gad_summary)
    require_csv_schema(gad_csv, gad.SUMMARY_FIELDS)

    morse_gmin.self_test()
    mg_rows = []
    for seed in morse_gmin.SEEDS:
        mg_rows.append(
            morse_gmin.anneal_line(
                f"m30r6,anneal,{seed},400000,true,12000,12000,0.4, -118.432844,0.3,0.1\n",
                "m30r6",
                "head-1",
                host="bench-a",
                machine=machine_a,
            )
        )
        mg_rows.append(
            morse_gmin.gmin_line(
                f"H2H best=-118.432844 evals_to_hit=15000 charged=15000 wall_s=0.5 pot_s=0.4\n",
                "m30r6",
                seed,
                400_000,
                -118.432844,
                "svn-1",
                host="bench-b",
                machine=machine_b,
            )
        )
    mg_seeds = tmp_path / "mg-seeds.csv"
    morse_gmin.write_rows(mg_seeds, mg_rows)
    require_csv_schema(mg_seeds, morse_gmin.FIELDS)
    mg_summary = morse_gmin.summarise(mg_rows, set())
    assert mg_summary[0]["machine_id"] == f"{'a' * 64};{'b' * 64}"
    assert mg_summary[0]["cpu_model"] == "CPU A;CPU B"

    morse_h2h.self_test()
    mh_saved = tmp_path / "mh-gmin.row"
    morse_h2h.write_per_seed_row(
        mh_saved,
        "gmin",
        {
            "system": "m38r14",
            "seed": "2000",
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "20000",
            "charged": "20000",
            "wall_s": "0.6",
            "best": "-144.321054",
            "pot_s": "0.5",
        },
        machine=machine_b,
        url="https://example.invalid/gmin",
        revision="svn-1",
        host="bench-b",
    )
    assert (
        morse_h2h.load_saved(
            mh_saved,
            "m38r14",
            "gmin",
            machine=machine_b,
            url="https://example.invalid/gmin",
            revision="svn-1",
        )
        is not None
    )
    assert (
        morse_h2h.load_saved(
            mh_saved,
            "m38r14",
            "gmin",
            machine=machine_b,
            url="https://example.invalid/gmin",
            revision="svn-2",
        )
        is None
    )
    mh_loaded = morse_h2h.load_saved(
        mh_saved,
        "m38r14",
        "gmin",
        machine=machine_b,
        url="https://example.invalid/gmin",
        revision="svn-1",
    )
    mh_summary = morse_h2h.summarise(
        [mh_loaded], (2000,), "https://example.invalid/gmin", "svn-1", "head-1", machine_a
    )
    assert mh_summary[0]["machine_id"] == "b" * 64
    assert mh_summary[0]["cpu_model"] == "CPU B"
    mh_csv = tmp_path / "mh-summary.csv"
    morse_h2h.write_summary(mh_csv, mh_summary)
    require_csv_schema(mh_csv, morse_h2h.SUMMARY_FIELDS)

    mp_rows = []
    for seed in morse_panel.SEEDS:
        mp_rows.append(
            {
                "system": "m55r6",
                "code": "anneal",
                "seed": str(seed),
                "budget": "1000000",
                "hit": "1",
                "evals_to_hit": "30000",
                "charged": "30000",
                "wall_s": "1.0",
                "best": "-250.286609",
                "pot_s": "0.8",
                "outside_s": "0.2",
                "code_revision": "head-1",
                "host": "bench-a",
                **machine_a,
            }
        )
        mp_rows.append(
            {
                "system": "m55r6",
                "code": "gmin",
                "seed": str(seed),
                "budget": "1000000",
                "hit": "1",
                "evals_to_hit": "40000",
                "charged": "40000",
                "wall_s": "1.2",
                "best": "-250.286609",
                "pot_s": "1.0",
                "outside_s": "0.2",
                "code_revision": "svn-1",
                "host": "bench-b",
                **machine_b,
            }
        )
    mp_seeds = tmp_path / "mp-seeds.csv"
    morse_panel.write_rows(mp_seeds, mp_rows)
    require_csv_schema(mp_seeds, morse_panel.ROW_FIELDS)
    mp_summary = morse_panel.summarise(mp_rows, set())
    assert mp_summary[0]["machine_id"] == f"{'a' * 64};{'b' * 64}"
    assert mp_summary[0]["cpu_model"] == "CPU A;CPU B"
