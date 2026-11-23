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
    validate_saved_row_identity,
)

ACTIVE_ROW_WRITERS = (
    "scripts/ase_nonlj/run_one.py",
    "scripts/ase_nonlj/summarize.py",
    "scripts/ase_panel/run_one.py",
    "scripts/ase_panel/summarize.py",
    "scripts/au55_scale/run_panel.py",
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
    "scripts/morse14_h2h/run_panel.py",
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

    def make_row(
        code: str,
        seed: int,
        code_revision: str,
        machine_id: str,
        cpu_model: str,
        gmin_build_key: str | None = None,
    ):
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
            "gmin_build_key": (
                ("c" * 64 if code == "gmin" else "")
                if gmin_build_key is None
                else gmin_build_key
            ),
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

    mixed_key_rows = [
        make_row("anneal", seed, "head-rev", "a" * 64, "CPU A") for seed in panel.SEEDS
    ] + [
        make_row(
            "gmin",
            seed,
            "svn-100",
            "b" * 64,
            "CPU B",
            gmin_build_key="c" * 64 if seed != panel.SEEDS[-1] else "d" * 64,
        )
        for seed in panel.SEEDS
    ]
    with pytest.raises((RuntimeError, ValueError), match="gmin_build_key"):
        panel.panel_groups(mixed_key_rows, "head-rev")
    with pytest.raises((RuntimeError, ValueError), match="gmin_build_key"):
        losses2.selected(mixed_key_rows, "lj38", "gmin", "head-rev", "base-rev")

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
        "gmin_build_key",
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
                    "gmin_build_key": "c" * 64,
                    "host": "remote-host",
                    "machine_id": "b" * 64,
                    "cpu_model": "Remote CPU",
                }
            )

    reused = budget_mod.load_reuse(gmin_csv, "svn-remote", "c" * 64)
    assert len(reused) == len(budget_mod.SEEDS)
    with pytest.raises(ValueError, match="code_revision"):
        budget_mod.load_reuse(gmin_csv, "svn-other", "c" * 64)
    with pytest.raises(ValueError, match="gmin_build_key"):
        budget_mod.load_reuse(gmin_csv, "svn-remote", "d" * 64)

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
            gmin_build_key="c" * 64,
            host="local-host",
            machine=local_machine,
            reused=True,
        )
        assert gmin_row["host"] == "remote-host"
        assert gmin_row["machine_id"] == "b" * 64
        assert gmin_row["cpu_model"] == "Remote CPU"
        assert gmin_row["code_revision"] == "svn-remote"
        assert gmin_row["gmin_build_key"] == "c" * 64
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
            gmin_build_key="c" * 64,
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
            2000,
            "anneal",
            gad.BUDGET,
            code_revision="head-1",
            machine=machine_a,
        )
        is not None
    )
    assert (
        gad.saved_rerun(
            gad_row_path,
            38,
            2000,
            "anneal",
            gad.BUDGET,
            code_revision="head-2",
            machine=machine_a,
        )
        is None
    )
    for bad_args, match in (
        ((31, 2000, "anneal", gad.BUDGET), "saved row system mismatch"),
        ((38, 2001, "anneal", gad.BUDGET), "saved row seed mismatch"),
        ((38, 2000, "gmin", gad.BUDGET), "saved row arm mismatch"),
        ((38, 2000, "anneal", 100_000), "saved row budget mismatch"),
    ):
        with pytest.raises(ValueError, match=match):
            gad.saved_rerun(
                gad_row_path,
                *bad_args,
                code_revision="head-1",
                machine=machine_a,
            )
    gad_rec = gad.saved_rerun(
        gad_row_path,
        38,
        2000,
        "anneal",
        gad.BUDGET,
        code_revision="head-1",
        machine=machine_a,
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
                gmin_build_key="c" * 64,
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
        gmin_build_key="c" * 64,
        host="bench-b",
    )
    assert (
        morse_h2h.load_saved(
            mh_saved,
            "m38r14",
            2000,
            "gmin",
            400_000,
            machine=machine_b,
            url="https://example.invalid/gmin",
            revision="svn-1",
            gmin_build_key="c" * 64,
        )
        is not None
    )
    with pytest.raises(ValueError, match="code_revision"):
        morse_h2h.load_saved(
            mh_saved,
            "m38r14",
            2000,
            "gmin",
            400_000,
            machine=machine_b,
            url="https://example.invalid/gmin",
            revision="svn-2",
            gmin_build_key="c" * 64,
        )
    with pytest.raises(ValueError, match="gmin_build_key"):
        morse_h2h.load_saved(
            mh_saved,
            "m38r14",
            2000,
            "gmin",
            400_000,
            machine=machine_b,
            url="https://example.invalid/gmin",
            revision="svn-1",
            gmin_build_key="d" * 64,
        )
    for bad_args, match in (
        (("m55r14", 2000, "gmin", 400_000), "saved row system mismatch"),
        (("m38r14", 2001, "gmin", 400_000), "saved row seed mismatch"),
        (("m38r14", 2000, "anneal", 400_000), "saved row arm mismatch"),
        (("m38r14", 2000, "gmin", 1_000_000), "saved row budget mismatch"),
    ):
        with pytest.raises(ValueError, match=match):
            morse_h2h.load_saved(
                mh_saved,
                *bad_args,
                machine=machine_b,
                url="https://example.invalid/gmin",
                revision="svn-1",
                gmin_build_key="c" * 64,
            )
    mh_loaded = morse_h2h.load_saved(
        mh_saved,
        "m38r14",
        2000,
        "gmin",
        400_000,
        machine=machine_b,
        url="https://example.invalid/gmin",
        revision="svn-1",
        gmin_build_key="c" * 64,
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
                "gmin_build_key": "",
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
                "gmin_build_key": "c" * 64,
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


def test_provenance_runners_and_summaries_run_without_scipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)

    ablation = _load_module("generic_alloy_ablation_noscipy", "scripts/generic_alloy/run_ablation.py")
    morse_panel = _load_module("morse_panel_noscipy", "scripts/morse_panel/run_panel.py")

    machine_a = {"machine_id": "a" * 64, "cpu_model": "CPU A"}
    machine_b = {"machine_id": "b" * 64, "cpu_model": "CPU B"}
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
                "gmin_build_key": "",
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
                "hit": "0",
                "evals_to_hit": "1000000",
                "charged": "1000000",
                "wall_s": "2.0",
                "best": "-249.000000",
                "pot_s": "1.6",
                "outside_s": "0.4",
                "code_revision": "svn-1",
                "gmin_build_key": "c" * 64,
                "host": "bench-b",
                **machine_b,
            }
        )
    mp_summary = morse_panel.summarise(mp_rows, set())
    assert mp_summary[0]["machine_id"] == f"{'a' * 64};{'b' * 64}"
    assert mp_summary[0]["cpu_model"] == "CPU A;CPU B"
    assert 0.0 <= float(mp_summary[0]["fisher_p"]) <= 1.0

    ablation_rows = [
        {
            "campaign": "priors-off",
            "system": "cu6au32",
            "arm": "develop",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.5",
            "best": "-130.79994836",
            "pot_s": "0.4",
            "outside_s": "0.1",
            "code_revision": "rev-1",
            "host": "bench",
            **machine_a,
        }
        for seed in ablation.SEEDS[:2]
    ] + [
        {
            "campaign": "priors-off",
            "system": "cu6au32",
            "arm": "descent",
            "seed": str(seed),
            "budget": "400000",
            "hit": "0",
            "evals_to_hit": "400000",
            "charged": "400000",
            "wall_s": "1.5",
            "best": "-129.0",
            "pot_s": "1.2",
            "outside_s": "0.3",
            "code_revision": "rev-2",
            "host": "bench",
            **machine_b,
        }
        for seed in ablation.SEEDS[:2]
    ]
    ab_summary = ablation.summarise("priors-off", ablation_rows)
    assert len(ab_summary) == 2
    assert 0.0 <= float(ab_summary[1]["fisher_p"]) <= 1.0


def test_validate_saved_row_identity_requires_every_arm_field_to_agree(tmp_path):
    row_path = tmp_path / "anneal-2000.row"
    base = {
        "system": "lj38",
        "code": "anneal",
        "arm": "anneal",
        "seed": "2000",
        "budget": "400000",
    }
    validate_saved_row_identity(
        base,
        path=row_path,
        system="lj38",
        arm="anneal",
        seed=2000,
        budget=400_000,
    )

    for bad_row, expected_tokens in (
        ({**base, "code": "anneal", "arm": "gmin"}, ("arm", "arm='gmin'")),
        ({**base, "code": "gmin", "arm": "anneal"}, ("arm", "code", "code='gmin'")),
        ({**base, "code": "anneal", "arm": ""}, ("arm", "arm=''")),
        ({**base, "code": "", "arm": "anneal"}, ("arm", "code", "code=''")),
        ({**base, "code": "anneal", "arm": None}, ("arm", "arm=''")),
        ({"system": "lj38", "seed": "2000", "budget": "400000"}, ("arm",)),
    ):
        with pytest.raises(ValueError) as exc_info:
            validate_saved_row_identity(
                bad_row,
                path=row_path,
                system="lj38",
                arm="anneal",
                seed=2000,
                budget=400_000,
            )
        msg = str(exc_info.value)
        assert str(row_path) in msg
        for token in expected_tokens:
            assert token in msg


EIGHT_VALIDATED_RUNNERS = (
    "gmin_h2h",
    "morse_h2h",
    "gad_published",
    "covalent_h2h",
    "growpal_h2h",
    "growpal_h2h_v2",
    "metal_h2h",
    "metal_darby",
)

SAVED_ROW_MUTATIONS = (
    "system",
    "native_arm",
    "code_matches_arm_disagrees",
    "arm_matches_code_disagrees",
    "seed",
    "budget",
)


def _write_dict_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)
    return path


def _mutate_row_for_field(
    valid_row: dict[str, str],
    *,
    native_arm_key: str,
    expected_arm: str,
    wrong_arm: str,
    wrong_system: str,
    wrong_seed: int,
    wrong_budget: int,
    mutation: str,
) -> tuple[dict[str, str], tuple[str, ...]]:
    if mutation == "system":
        return {**valid_row, "system": wrong_system}, ("system",)
    if mutation == "native_arm":
        return {**valid_row, native_arm_key: wrong_arm}, ("arm", native_arm_key)
    if mutation == "code_matches_arm_disagrees":
        return {**valid_row, "code": expected_arm, "arm": wrong_arm}, ("arm",)
    if mutation == "arm_matches_code_disagrees":
        return {**valid_row, "code": wrong_arm, "arm": expected_arm}, ("arm", "code")
    if mutation == "seed":
        return {**valid_row, "seed": str(wrong_seed)}, ("seed",)
    if mutation == "budget":
        return {**valid_row, "budget": str(wrong_budget)}, ("budget",)
    raise AssertionError(f"unknown mutation {mutation!r}")


def _runner_saved_row_spec(runner_id: str, machine: dict[str, str], seed: int = 2000):
    if runner_id == "gmin_h2h":
        mod = _load_module("rej_gmin_h2h", "scripts/gmin_h2h/run_campaign.py")
        row = {
            "system": "lj98",
            "code": "anneal",
            "seed": str(seed),
            "budget": "4000000",
            "hit": "1",
            "evals_to_hit": "12000",
            "charged": "12000",
            "gradient_passes": "",
            "energy_only_entries": "",
            "wall_s": "0.500000",
            "best": "-543.665361",
            "pot_s": "0.400000",
            "gmin_url": "",
            "gmin_revision": "",
            "gmin_build_key": "",
            "anneal_head": "head-1",
            "code_revision": "head-1",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "code",
            "anneal",
            "gmin",
            "lj104",
            seed + 1,
            800_000,
            lambda p: mod.load_saved(
                p,
                "lj98",
                seed,
                "anneal",
                4_000_000,
                machine["machine_id"],
                machine["cpu_model"],
                code_revision="head-1",
                gmin_url="https://example.invalid/gmin.tar.gz",
                gmin_build_key="b" * 64,
            ),
        )
    if runner_id == "morse_h2h":
        mod = _load_module("rej_morse_h2h", "scripts/morse_h2h/run_campaign.py")
        row = {
            "system": "m38r14",
            "code": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "15000",
            "charged": "15000",
            "wall_s": "0.600000",
            "best": "-144.321054",
            "pot_s": "0.500000",
            "outside_s": "0.100000",
            "gmin_url": "",
            "gmin_revision": "",
            "anneal_head": "head-1",
            "code_revision": "head-1",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "code",
            "anneal",
            "gmin",
            "m55r14",
            seed + 1,
            1_000_000,
            lambda p: mod.load_saved(
                p,
                "m38r14",
                seed,
                "anneal",
                400_000,
                machine=machine,
                url="https://example.invalid/gmin.tar.gz",
                revision="svn-1",
                head="head-1",
            ),
        )
    if runner_id == "gad_published":
        mod = _load_module("rej_gad_published", "scripts/gad_published/run_campaign.py")
        row = {
            "system": "lj38",
            "code": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.500000",
            "best": "-173.928427",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "code_revision": "head-1",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "code",
            "anneal",
            "gmin",
            "lj31",
            seed + 1,
            100_000,
            lambda p: mod.saved_rerun(
                p,
                38,
                seed,
                "anneal",
                mod.BUDGET,
                code_revision="head-1",
                machine=machine,
            ),
        )
    if runner_id == "covalent_h2h":
        mod = _load_module("rej_covalent_h2h", "scripts/covalent_h2h/run_campaign.py")
        row = {
            "system": "sw10",
            "code": "anneal",
            "seed": str(seed),
            "budget": "100000",
            "hit": "1",
            "evals_to_hit": "5000",
            "charged": "5000",
            "wall_s": "0.300000",
            "best": "-20.000000",
            "pot_s": "0.200000",
            "outside_s": "0.100000",
            "code_revision": "head-1",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "code",
            "anneal",
            "gmin",
            "sw13",
            seed + 1,
            400_000,
            lambda p: mod.load_saved(
                p,
                "sw10",
                seed,
                "anneal",
                100_000,
                machine=machine,
            ),
        )
    if runner_id == "growpal_h2h":
        mod = _load_module("rej_growpal_h2h", "scripts/growpal_h2h/run_campaign.py")
        row = {
            "system": "lj38",
            "arm": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "12000",
            "charged": "12000",
            "wall_s": "0.500000",
            "pot_s": "0.400000",
            "inside_us": "33.3333",
            "outside_us": "8.3333",
            "best": "-173.928427",
            "stop": "hit",
            "reached": "1",
            "code_revision": "head-1",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "arm",
            "anneal",
            "growpal",
            "lj75",
            seed + 1,
            4_000_000,
            lambda p: mod.load_saved(
                p,
                "lj38",
                seed,
                "anneal",
                400_000,
                machine=machine,
            ),
        )
    if runner_id == "growpal_h2h_v2":
        mod = _load_module("rej_growpal_h2h_v2", "scripts/growpal_h2h_v2/run_campaign.py")
        row = {
            "system": "lj38",
            "arm": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "12000",
            "charged": "12000",
            "wall_s": "0.500000",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "inside_us": "33.3333",
            "outside_us": "8.3333",
            "process_s": "0.500000",
            "best": "-173.928427",
            "stop": "hit",
            "reached": "1",
            "code_revision": "head-1",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "arm",
            "anneal",
            "growpal",
            "lj55",
            seed + 1,
            1_000_000,
            lambda p: mod.load_saved(p, "lj38", seed, "anneal", 400_000),
        )
    if runner_id == "metal_h2h":
        mod = _load_module("rej_metal_h2h", "scripts/metal_h2h/run_campaign.py")
        row = {
            "system": "zn38",
            "code": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.500000",
            "best": "-65.000000",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "pot_per_eval_s": "0.000040000",
            "outside_per_eval_s": "0.000010000",
        }
        return (
            mod,
            row,
            "code",
            "anneal",
            "ase",
            "zn55",
            seed + 1,
            900_000,
            lambda p: mod.load_saved(p, "zn38", seed, "anneal", 400_000),
        )
    if runner_id == "metal_darby":
        mod = _load_module("rej_metal_darby", "scripts/metal_h2h/run_darby.py")
        row = {
            "system": "cu19au19",
            "code": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.500000",
            "best": "-122.991484",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "code_revision": "head-1",
            "gmin_build_key": "",
            "host": "bench",
            **machine,
        }
        return (
            mod,
            row,
            "code",
            "anneal",
            "gmin",
            "au38",
            seed + 1,
            1_000_000,
            lambda p: mod.load_saved(
                p,
                "cu19au19",
                seed,
                "anneal",
                400_000,
                code_revision="head-1",
                machine=machine,
            ),
        )
    raise AssertionError(f"unknown runner {runner_id!r}")


@pytest.mark.parametrize("runner_id", EIGHT_VALIDATED_RUNNERS)
@pytest.mark.parametrize("mutation", SAVED_ROW_MUTATIONS)
def test_all_eight_runners_reject_bad_saved_row_for_every_validated_field(
    tmp_path, runner_id, mutation
):
    machine = {"machine_id": "a" * 64, "cpu_model": "Example CPU"}
    (
        _mod,
        valid_row,
        native_arm_key,
        expected_arm,
        wrong_arm,
        wrong_system,
        wrong_seed,
        wrong_budget,
        loader,
    ) = _runner_saved_row_spec(runner_id, machine)

    good_path = tmp_path / runner_id / "valid.row"
    _write_dict_csv(good_path, valid_row)
    assert loader(good_path) is not None

    bad_row, expected_tokens = _mutate_row_for_field(
        valid_row,
        native_arm_key=native_arm_key,
        expected_arm=expected_arm,
        wrong_arm=wrong_arm,
        wrong_system=wrong_system,
        wrong_seed=wrong_seed,
        wrong_budget=wrong_budget,
        mutation=mutation,
    )
    bad_path = tmp_path / runner_id / f"bad-{mutation}.row"
    _write_dict_csv(bad_path, bad_row)
    with pytest.raises(ValueError) as exc_info:
        loader(bad_path)
    msg = str(exc_info.value)
    assert str(bad_path) in msg
    for token in expected_tokens:
        assert token in msg


@pytest.mark.parametrize("runner_id", EIGHT_VALIDATED_RUNNERS)
@pytest.mark.parametrize("mutation", SAVED_ROW_MUTATIONS)
def test_all_eight_runners_main_rejects_bad_saved_row_for_every_validated_field(
    tmp_path, monkeypatch, runner_id, mutation
):
    import os
    import types

    from gmin_pin import REVISION, TARBALL_SHA256
    from machine_provenance import machine_columns

    real_uname = os.uname()
    fake_uname = os.uname_result(
        (
            real_uname.sysname,
            "bench",
            real_uname.release,
            real_uname.version,
            real_uname.machine,
        )
    )
    monkeypatch.setattr(os, "uname", lambda: fake_uname)
    machine = machine_columns()

    work = tmp_path / runner_id / mutation
    work.mkdir(parents=True, exist_ok=True)
    head_file = work / "head.txt"
    head_file.write_text("head-1\n")
    noop_bin = _write_executable(work / "noop", "exit 1\n")

    if runner_id == "covalent_h2h":
        covalent_preview = _load_module(
            f"preview_{runner_id}_{mutation}", "scripts/covalent_h2h/run_campaign.py"
        )
        target_seed = covalent_preview.schedule(
            ("sw10",),
            covalent_preview.SEEDS,
            lambda _name: ("anneal",),
            covalent_preview.SCHEDULE_SEED,
        )[0][1]
    else:
        target_seed = 2000

    (
        mod,
        valid_row,
        native_arm_key,
        expected_arm,
        wrong_arm,
        wrong_system,
        wrong_seed,
        wrong_budget,
        _loader,
    ) = _runner_saved_row_spec(runner_id, machine, seed=target_seed)

    gmin_source_text = (
        f"url=https://example.invalid/gmin.tar.gz\n"
        f"revision={REVISION}\n"
        f"svnrev={REVISION}\n"
        f"tarball_sha256={TARBALL_SHA256}\n"
        f"build_key={'b' * 64}\n"
    )
    if runner_id == "gmin_h2h":
        binary = _write_executable(
            work / "h2h_bin",
            'if [ "$1" = "budget" ]; then printf "4000000\\n-543.665361\\n"; exit 0; fi\nexit 1\n',
        )
        source = work / "source.txt"
        source.write_text(gmin_source_text)
        svnrev = work / "GMIN.svnrev"
        svnrev.write_text(f"{REVISION}\n")
        data = work / "data"
        data.write_text("SLOPPYCONV 1.0D-4\n")
        runs = work / "camp" / "runs"
        starts = work / "starts"
        coord = starts / "lj98" / "seed-2000.xyz"
        coord.parent.mkdir(parents=True, exist_ok=True)
        coord.write_text("98\n\n")
        saved_path = runs / "lj98" / "anneal-2000.row"
        argv = [
            "run_campaign.py",
            "--binary",
            str(binary),
            "--gmin",
            str(noop_bin),
            "--data",
            str(data),
            "--source",
            str(source),
            "--svnrev",
            str(svnrev),
            "--head-file",
            str(head_file),
            "--runs",
            str(runs),
            "--starts",
            str(starts),
            "--size",
            "98",
            "--seed",
            "2000",
            "--arm",
            "anneal",
        ]
    elif runner_id == "morse_h2h":
        binary = _write_executable(
            work / "morse_bin",
            'if [ "$1" = "budget" ]; then printf "400000\\n-144.321054\\n14.0\\n"; exit 0; fi\nexit 1\n',
        )
        source = work / "source.txt"
        source.write_text(gmin_source_text)
        svnrev = work / "GMIN.svnrev"
        svnrev.write_text(f"{REVISION}\n")
        data = work / "data"
        data.write_text("SLOPPYCONV 1.0D-4\n")
        runs = work / "camp" / "runs"
        starts = work / "starts"
        coord = starts / "m38r14" / "seed-2000.xyz"
        coord.parent.mkdir(parents=True, exist_ok=True)
        coord.write_text("38\n\n")
        saved_path = runs / "m38r14" / "anneal-2000.row"
        argv = [
            "run_campaign.py",
            "--binary",
            str(binary),
            "--gmin",
            str(noop_bin),
            "--data",
            str(data),
            "--source",
            str(source),
            "--svnrev",
            str(svnrev),
            "--head-file",
            str(head_file),
            "--runs",
            str(runs),
            "--starts",
            str(starts),
            "--system",
            "m38r14",
            "--seed",
            "2000",
            "--arm",
            "anneal",
        ]
    elif runner_id == "gad_published":
        monkeypatch.setattr(mod, "gad_rows", lambda _zip: [])
        source = work / "source.txt"
        source.write_text(gmin_source_text)
        svnrev = work / "GMIN.svnrev"
        svnrev.write_text(f"{REVISION}\n")
        data = work / "data"
        data.write_text("SLOPPYCONV 1.0D-4\n")
        zip_path = work / "deposit.zip"
        zip_path.write_bytes(b"")
        runs = work / "camp" / "runs"
        saved_path = runs / "lj38" / "anneal-2000.row"
        argv = [
            "run_campaign.py",
            "--binary",
            str(noop_bin),
            "--gmin",
            str(noop_bin),
            "--data",
            str(data),
            "--zip",
            str(zip_path),
            "--source",
            str(source),
            "--svnrev",
            str(svnrev),
            "--head-file",
            str(head_file),
            "--runs",
            str(runs),
            "--size",
            "38",
            "--seed",
            "2000",
            "--arm",
            "anneal",
        ]
    elif runner_id == "covalent_h2h":
        binary = _write_executable(
            work / "cov_bin",
            'if [ "$1" = "budget" ]; then printf "100000\\n-20.0\\n10\\n"; exit 0; fi\n'
            'if [ "$1" = "scales" ]; then printf "1.0\\n1.0\\n2.5\\n"; exit 0; fi\n'
            "exit 1\n",
        )
        source = work / "source.txt"
        source.write_text(gmin_source_text)
        svnrev = work / "GMIN.svnrev"
        svnrev.write_text(f"{REVISION}\n")
        data = work / "data"
        data.write_text("SLOPPYCONV 5.0D-3\nTIGHTCONV 1.0D-3\nSW\n")
        runs = work / "camp" / "runs"
        starts = work / "starts"
        for seed in mod.SEEDS:
            coord = starts / "sw10" / f"seed-{seed}.xyz"
            coord.parent.mkdir(parents=True, exist_ok=True)
            coord.write_text("10\n\n")
            _write_dict_csv(
                runs / "sw10" / f"anneal-{seed}.row",
                {**valid_row, "seed": str(seed)},
            )
        saved_path = runs / "sw10" / f"anneal-{target_seed}.row"
        argv = [
            "run_campaign.py",
            "--binary",
            str(binary),
            "--library",
            str(work / "lib.so"),
            "--gmin",
            str(noop_bin),
            "--data",
            str(data),
            "--gmin-tersoff",
            str(noop_bin),
            "--carbon-data",
            str(data),
            "--ase",
            str(noop_bin),
            "--source",
            str(source),
            "--svnrev",
            str(svnrev),
            "--head-file",
            str(head_file),
            "--runs",
            str(runs),
            "--starts",
            str(starts),
            "--rows",
            str(work / "rows.csv"),
            "--summary",
            str(work / "summary.csv"),
            "--system",
            "sw10",
            "--code",
            "anneal",
        ]
    elif runner_id == "growpal_h2h":
        monkeypatch.setitem(
            sys.modules,
            "growpal_arm",
            types.SimpleNamespace(assert_budget_stops_lbfgs=lambda: None),
        )
        binary = _write_executable(
            work / "gp_bin",
            'if [ "$1" = "budget" ]; then printf "400000\\n-173.928427\\n"; exit 0; fi\nexit 1\n',
        )
        source = work / "source.txt"
        source.write_text(
            "growpal_version=0.1.0\naegon_version=0.1.0\ngrowpal_source=sdist\n"
        )
        seeds_xyz = work / "seeds.xyz"
        seeds_xyz.write_text("")
        runs = work / "camp" / "runs"
        rows_dir = work / "camp" / "rows"
        saved_path = rows_dir / "lj38" / "anneal-2000.row"
        argv = [
            "run_campaign.py",
            "--binary",
            str(binary),
            "--source",
            str(source),
            "--seeds-xyz",
            str(seeds_xyz),
            "--head-file",
            str(head_file),
            "--runs",
            str(runs),
            "--rows",
            str(rows_dir),
            "--size",
            "38",
            "--seed",
            "2000",
            "--arm",
            "anneal",
        ]
    elif runner_id == "growpal_h2h_v2":
        bin_dir = work / "bin"
        _write_executable(
            bin_dir / "h2h_gmin_proof",
            'if [ "$1" = "budget" ]; then printf "400000\\n-173.928427\\n"; exit 0; fi\nexit 1\n',
        )
        _write_executable(bin_dir / "h2h_morse", "exit 1\n")
        py_stub = _write_executable(work / "py_stub", "exit 0\n")
        source = work / "source.txt"
        source.write_text(
            "growpal_version=0.1.0\n"
            "growpal_sdist_sha256=aa\n"
            "growpal_commit=bb\n"
            "aegon_version=0.1.0\n"
            "aegon_sdist_sha256=cc\n"
            "ase_version=3.23.0\n"
        )
        seeds_xyz = work / "seeds.xyz"
        seeds_xyz.write_text("")
        runs = work / "runs"
        out_dir = work / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "identity.csv").write_text(",".join(mod.IDENTITY_FIELDS) + "\n")
        saved_path = out_dir / "rows" / "lj38" / "anneal-2000.row"
        argv = [
            "run_campaign.py",
            "--bin-dir",
            str(bin_dir),
            "--python",
            str(py_stub),
            "--source",
            str(source),
            "--seeds-xyz",
            str(seeds_xyz),
            "--head-file",
            str(head_file),
            "--runs",
            str(runs),
            "--out",
            str(out_dir),
            "--system",
            "lj38",
            "--seed",
            "2000",
            "--arm",
            "anneal",
        ]
    elif runner_id == "metal_h2h":
        binary = _write_executable(
            work / "metal_bin",
            'if [ "$1" = "budget" ]; then printf "400000\\n-65.0\\n"; exit 0; fi\nexit 1\n',
        )
        runs = work / "camp" / "runs"
        logs = work / "camp" / "logs"
        saved_path = runs / "zn38" / "anneal-2000.row"
        argv = [
            "run_campaign.py",
            "--binary",
            str(binary),
            "--ga",
            str(noop_bin),
            "--runs",
            str(runs),
            "--logs",
            str(logs),
            "--head-file",
            str(head_file),
            "--system",
            "zn38",
            "--seed",
            "2000",
            "--arm",
            "anneal",
        ]
    elif runner_id == "metal_darby":
        monkeypatch.setattr(Path, "cwd", classmethod(lambda _cls: Path("/work")))
        monkeypatch.setattr(mod, "check_potentials", lambda *_args: None)
        source = work / "source.txt"
        source.write_text(gmin_source_text)
        svnrev = work / "GMIN.svnrev"
        svnrev.write_text(f"{REVISION}\n")
        runs = work / "runs"
        logs = work / "logs"
        coord = runs / "cu19au19" / "start-2000.xyz"
        coord.parent.mkdir(parents=True, exist_ok=True)
        coord.write_text("38\n\n")
        saved_path = runs / "cu19au19" / "anneal-2000.row"
        argv = [
            "--binary",
            str(noop_bin),
            "--gmin",
            str(noop_bin),
            "--fixtures",
            str(work / "fixtures"),
            "--runs",
            str(runs),
            "--logs",
            str(logs),
            "--summary",
            str(work / "summary.csv"),
            "--source",
            str(source),
            "--svnrev",
            str(svnrev),
            "--head-file",
            str(head_file),
            "--only",
            "cu19au19",
            "2000",
            "anneal",
        ]
    else:
        raise AssertionError(f"unknown runner {runner_id!r}")

    _write_dict_csv(saved_path, valid_row)
    assert mod.main(argv) == 0

    bad_row, expected_tokens = _mutate_row_for_field(
        valid_row,
        native_arm_key=native_arm_key,
        expected_arm=expected_arm,
        wrong_arm=wrong_arm,
        wrong_system=wrong_system,
        wrong_seed=wrong_seed,
        wrong_budget=wrong_budget,
        mutation=mutation,
    )
    _write_dict_csv(saved_path, bad_row)
    with pytest.raises(ValueError) as exc_info:
        mod.main(argv)
    msg = str(exc_info.value)
    assert str(saved_path) in msg
    for token in expected_tokens:
        assert token in msg


def test_native_headerless_and_h2h_row_formats_reject_every_bad_field(tmp_path):
    metal = _load_module("native_metal_h2h", "scripts/metal_h2h/run_campaign.py")
    darby = _load_module("native_metal_darby", "scripts/metal_h2h/run_darby.py")
    gmin_h2h = _load_module("native_gmin_h2h", "scripts/gmin_h2h/run_campaign.py")
    morse_h2h = _load_module("native_morse_h2h", "scripts/morse_h2h/run_campaign.py")
    gad = _load_module("native_gad_published", "scripts/gad_published/run_campaign.py")

    metal_ok = tmp_path / "metal-ok.row"
    metal_ok.write_text(
        "zn38,anneal,2000,400000,1,10000,10000,0.5,-65.0,0.4,0.1,4e-5,1e-5\n"
    )
    assert metal.load_saved(metal_ok, "zn38", 2000, "anneal", 400_000) is not None

    for idx, (bad_line, field_token) in enumerate(
        (
            ("zn55,anneal,2000,400000,1,10000,10000,0.5,-65.0,0.4,0.1,4e-5,1e-5\n", "system"),
            ("zn38,gmin,2000,400000,1,10000,10000,0.5,-65.0,0.4,0.1,4e-5,1e-5\n", "code"),
            ("zn38,anneal,2001,400000,1,10000,10000,0.5,-65.0,0.4,0.1,4e-5,1e-5\n", "seed"),
            ("zn38,anneal,2000,900000,1,10000,10000,0.5,-65.0,0.4,0.1,4e-5,1e-5\n", "budget"),
        )
    ):
        bad_path = tmp_path / f"metal-bad-{idx}.row"
        bad_path.write_text(bad_line)
        with pytest.raises(ValueError) as exc_info:
            metal.load_saved(bad_path, "zn38", 2000, "anneal", 400_000)
        assert field_token in str(exc_info.value)

    darby_gmin_ok = tmp_path / "darby-gmin-ok.row"
    darby_gmin_ok.write_text(
        "h2h_row system=cu19au19 code=gmin seed=2000 budget=400000 hit=1 "
        "evals_to_hit=10000 charged=10000 gradient_passes=10000 energy_only_entries=5 "
        f"best=-122.991484 wall_s=0.5 pot_s=0.4 code_revision=svn-1 gmin_build_key={'b' * 64}\n"
    )
    assert (
        darby.load_saved(
            darby_gmin_ok,
            "cu19au19",
            2000,
            "gmin",
            400_000,
            code_revision="svn-1",
            gmin_build_key="b" * 64,
        )
        is not None
    )
    darby_gmin_unkeyed = tmp_path / "darby-gmin-unkeyed.row"
    darby_gmin_unkeyed.write_text(
        "h2h_row system=cu19au19 code=gmin seed=2000 budget=400000 hit=1 "
        "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4 "
        "code_revision=svn-1\n"
    )
    with pytest.raises(ValueError, match="gmin_build_key"):
        darby.load_saved(
            darby_gmin_unkeyed,
            "cu19au19",
            2000,
            "gmin",
            400_000,
            code_revision="svn-1",
            gmin_build_key="b" * 64,
        )

    for idx, (bad_kv, field_token) in enumerate(
        (
            (
                "h2h_row system=au38 code=gmin seed=2000 budget=400000 hit=1 "
                "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4\n",
                "system",
            ),
            (
                "h2h_row system=cu19au19 code=gmin arm=anneal seed=2000 budget=400000 hit=1 "
                "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4\n",
                "arm",
            ),
            (
                "h2h_row system=cu19au19 arm=anneal seed=2000 budget=400000 hit=1 "
                "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4\n",
                "arm",
            ),
            (
                "h2h_row system=cu19au19 seed=2000 budget=400000 hit=1 "
                "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4\n",
                "arm",
            ),
            (
                "h2h_row system=cu19au19 code=gmin seed=2001 budget=400000 hit=1 "
                "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4\n",
                "seed",
            ),
            (
                "h2h_row system=cu19au19 code=gmin seed=2000 budget=900000 hit=1 "
                "evals_to_hit=10000 charged=10000 best=-122.991484 wall_s=0.5 pot_s=0.4\n",
                "budget",
            ),
        )
    ):
        bad_path = tmp_path / f"darby-bad-{idx}.row"
        bad_path.write_text(bad_kv)
        with pytest.raises(ValueError) as exc_info:
            darby.load_saved(
                bad_path,
                "cu19au19",
                2000,
                "gmin",
                400_000,
                code_revision="svn-1",
                gmin_build_key="b" * 64,
            )
        assert field_token in str(exc_info.value)

    contradictory_kv = (
        "h2h_row system=lj38 code=gmin arm=anneal seed=2000 budget=400000 hit=1 "
        "evals_to_hit=10000 charged=10000 best=-173.928427 wall_s=0.5 pot_s=0.4\n"
    )
    kv_path = tmp_path / "contradictory-kv.row"
    kv_path.write_text(contradictory_kv)
    machine = {"machine_id": "a" * 64, "cpu_model": "Example CPU"}
    with pytest.raises(ValueError, match="arm"):
        gmin_h2h.load_saved(
            kv_path,
            "lj38",
            2000,
            "gmin",
            400_000,
            "a" * 64,
            "Example CPU",
            code_revision="svn-1",
            gmin_build_key="b" * 64,
        )
    with pytest.raises(ValueError, match="arm"):
        morse_h2h.load_saved(
            kv_path, "lj38", 2000, "gmin", 400_000, machine=machine, revision="svn-1"
        )
    with pytest.raises(ValueError, match="arm"):
        gad.saved_rerun(kv_path, 38, 2000, "gmin", 400_000, code_revision="svn-1", machine=machine)


def test_au55_scale_and_morse14_h2h_write_machine_provenance_rows_and_summaries(
    tmp_path, monkeypatch
):
    import os
    import subprocess
    from gmin_pin import REVISION, TARBALL_SHA256

    au55 = _load_module("mp_au55_scale", "scripts/au55_scale/run_panel.py")
    morse14 = _load_module("mp_morse14_h2h", "scripts/morse14_h2h/run_panel.py")

    for mod in (au55, morse14):
        assert mod.ROW_FIELDS[-2:] == ["machine_id", "cpu_model"]

    machine_cur = {"machine_id": "a" * 64, "cpu_model": "Current Host CPU"}
    machine_stored = {"machine_id": "b" * 64, "cpu_model": "Stored GMIN CPU"}
    build_key = "c" * 64

    real_uname = os.uname()
    monkeypatch.setattr(
        os,
        "uname",
        lambda: os.uname_result(
            (
                real_uname.sysname,
                "bench",
                real_uname.release,
                real_uname.version,
                real_uname.machine,
            )
        ),
    )
    monkeypatch.setattr(au55, "machine_columns", lambda: dict(machine_cur))
    monkeypatch.setattr(
        au55,
        "with_machine_provenance",
        lambda row: {**row, **machine_cur},
    )
    monkeypatch.setattr(morse14, "machine_columns", lambda: dict(machine_cur))
    monkeypatch.setattr(au55, "SEEDS", [2000, 2001])
    monkeypatch.setattr(morse14, "SEEDS", [2000, 2001])

    fake_root = tmp_path / "repo"
    fake_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=fake_root, check=True)
    (fake_root / "README").write_text("fixture\n")
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    subprocess.run(["git", "add", "README"], cwd=fake_root, env=git_env, check=True)
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"],
        cwd=fake_root,
        env=git_env,
        check=True,
    )
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=fake_root, text=True
    ).strip()

    svnrev = tmp_path / "SVNREV"
    svnrev.write_text(f"{REVISION}\n")
    gmin_source = tmp_path / "gmin-source.txt"
    gmin_source.write_text(
        f"url=https://example.invalid/gmin.tar.bz2\n"
        f"revision={REVISION}\n"
        f"svnrev={REVISION}\n"
        f"tarball_sha256={TARBALL_SHA256}\n"
        f"build_key={build_key}\n"
    )

    # Seed stored GMIN archives for both panels on machine_stored.
    gupta_dir = fake_root / "bench" / "au55-gupta"
    gupta_dir.mkdir(parents=True)
    with (gupta_dir / "seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=au55.ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for seed in (2000, 2001):
            writer.writerow(
                {
                    "system": "au55",
                    "code": "gmin",
                    "seed": str(seed),
                    "budget": "1000000",
                    "hit": "0",
                    "evals_to_hit": "1000000",
                    "charged": "1000000",
                    "wall_s": "2.000000",
                    "best": "-190.50000000",
                    "pot_s": "1.800000",
                    "outside_s": "0.200000",
                    "code_revision": REVISION,
                    "gmin_build_key": build_key,
                    "host": "stored-host",
                    **machine_stored,
                }
            )

    monkeypatch.setattr(
        au55,
        "run_anneal",
        lambda _bin, seed: {
            "system": "au55",
            "code": "anneal",
            "seed": str(seed),
            "budget": "1000000",
            "hit": "1",
            "evals_to_hit": "50000",
            "charged": "50000",
            "wall_s": "1.000000",
            "best": "-191.03085280",
            "pot_s": "0.800000",
            "outside_s": "0.200000",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_panel.py",
            "--root",
            str(fake_root),
            "--svnrev",
            str(svnrev),
            "--gmin-source",
            str(gmin_source),
        ],
    )
    assert au55.main() == 0

    with (fake_root / "bench" / "au55-scale" / "seeds.csv").open() as handle:
        au55_reader = csv.DictReader(handle)
        assert au55_reader.fieldnames == au55.ROW_FIELDS
        au55_rows = list(au55_reader)
    anneal_au = [r for r in au55_rows if r["code"] == "anneal"]
    gmin_au = [r for r in au55_rows if r["code"] == "gmin"]
    assert all(
        r["code_revision"] == head_sha
        and r["machine_id"] == machine_cur["machine_id"]
        and r["cpu_model"] == machine_cur["cpu_model"]
        for r in anneal_au
    )
    assert all(
        r["code_revision"] == REVISION
        and r["machine_id"] == machine_stored["machine_id"]
        and r["cpu_model"] == machine_stored["cpu_model"]
        for r in gmin_au
    )
    with (fake_root / "bench" / "au55-scale" / "summary.csv").open() as handle:
        au55_sum = {r["arm"]: r for r in csv.DictReader(handle)}
    assert au55_sum["anneal"]["machine_id"] == machine_cur["machine_id"]
    assert au55_sum["gmin"]["machine_id"] == machine_stored["machine_id"]

    # Seed Morse archive and exercise morse14_h2h without SciPy.
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    morse_archive = fake_root / "bench" / "morse-h2h" / "seeds.csv"
    morse_archive.parent.mkdir(parents=True)
    with morse_archive.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=morse14.ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for sys_name, spec in morse14.SYSTEMS.items():
            for seed in (2000, 2001):
                writer.writerow(
                    {
                        "system": sys_name,
                        "code": "gmin",
                        "seed": str(seed),
                        "budget": str(spec["budget"]),
                        "hit": "0",
                        "evals_to_hit": str(spec["budget"]),
                        "charged": str(spec["budget"]),
                        "wall_s": "3.000000",
                        "best": f"{spec['reference'] + 1.0:.6f}",
                        "pot_s": "2.500000",
                        "outside_s": "0.500000",
                        "code_revision": REVISION,
                        "gmin_build_key": build_key,
                        "host": "stored-host",
                        **machine_stored,
                    }
                )
    fake_morse_bin = tmp_path / "h2h_morse"
    fake_morse_bin.write_text("#!/bin/sh\nexit 99\n")
    monkeypatch.setattr(morse14, "binary_path", lambda _root: fake_morse_bin)
    monkeypatch.setattr(
        morse14,
        "run_anneal",
        lambda _bin, sys_name, seed: {
            "system": sys_name,
            "code": "anneal",
            "seed": str(seed),
            "budget": str(morse14.SYSTEMS[sys_name]["budget"]),
            "hit": "1",
            "evals_to_hit": "40000",
            "charged": "40000",
            "wall_s": "1.200000",
            "best": f"{morse14.SYSTEMS[sys_name]['reference']:.6f}",
            "pot_s": "0.900000",
            "outside_s": "0.300000",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_panel.py",
            "--root",
            str(fake_root),
            "--svnrev",
            str(svnrev),
            "--gmin-source",
            str(gmin_source),
        ],
    )
    assert morse14.main() == 0

    with (fake_root / "bench" / "morse14-h2h" / "seeds.csv").open() as handle:
        m14_reader = csv.DictReader(handle)
        assert m14_reader.fieldnames == morse14.ROW_FIELDS
        m14_rows = list(m14_reader)
    assert all(
        r["machine_id"] == machine_cur["machine_id"]
        and r["cpu_model"] == machine_cur["cpu_model"]
        for r in m14_rows
        if r["code"] == "anneal"
    )
    assert all(
        r["machine_id"] == machine_stored["machine_id"]
        and r["cpu_model"] == machine_stored["cpu_model"]
        for r in m14_rows
        if r["code"] == "gmin"
    )
    with (fake_root / "bench" / "morse14-h2h" / "summary.csv").open() as handle:
        m14_sum = list(csv.DictReader(handle))
    assert all(
        r["machine_id"] == f"{machine_cur['machine_id']};{machine_stored['machine_id']}"
        and r["cpu_model"] == f"{machine_cur['cpu_model']};{machine_stored['cpu_model']}"
        for r in m14_sum
    )


def test_metal_h2h_persists_row_provenance_and_marks_legacy_reused_rows_unknown(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    metal = _load_module("metal_h2h_row_prov_mod", "scripts/metal_h2h/run_campaign.py")

    def fake_campaign_run(
        work: Path,
        *,
        head: str,
        machine: dict[str, str],
        seeds: tuple[int, ...],
        summary_path: Path,
    ) -> list[dict[str, str]]:
        head_file = work / "anneal-head"
        head_file.write_text(head + "\n")
        monkeypatch.setattr(metal, "binary_budget", lambda _bin, _sys: (20000, -10.0))

        def fake_run_arm(
            _bin, _ga, system, seed, budget, arm, log, _scratch
        ) -> dict[str, str]:
            line = (
                f"{system},{arm},{seed},{budget},1,500,500,"
                "0.250,-10.0,0.200,0.050,4e-4,1e-4"
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(line + "\n")
            return metal.parse_row(line)

        monkeypatch.setattr(metal, "run_arm", fake_run_arm)
        monkeypatch.setattr(metal, "machine_columns", lambda: dict(machine))
        monkeypatch.setattr(
            metal.os, "uname", lambda: SimpleNamespace(nodename="bench")
        )
        seed_flags: list[str] = []
        for seed in seeds:
            seed_flags.extend(["--seed", str(seed)])
        assert (
            metal.main(
                [
                    "run_campaign.py",
                    "--binary",
                    str(work / "h2h_metal"),
                    "--ga",
                    str(work / "ase_ga.py"),
                    "--runs",
                    str(work / "runs"),
                    "--logs",
                    str(work / "logs"),
                    "--head-file",
                    str(head_file),
                    "--summary",
                    str(summary_path),
                    "--system",
                    "zn38",
                    "--arm",
                    "anneal",
                    *seed_flags,
                ]
            )
            == 0
        )
        with summary_path.open() as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == metal.SUMMARY_FIELDS
            return list(reader)

    work_measure = tmp_path / "measure"
    work_measure.mkdir()
    measured_machine = {"machine_id": "1" * 64, "cpu_model": "Measured CPU"}
    summary_measured = fake_campaign_run(
        work_measure,
        head="rev-measured-111",
        machine=measured_machine,
        seeds=(0,),
        summary_path=work_measure / "summary-initial.csv",
    )

    saved_seed0 = work_measure / "runs" / "zn38" / "anneal-0.row"
    with saved_seed0.open() as handle:
        saved_reader = csv.DictReader(handle)
        assert saved_reader.fieldnames == metal.ROW_FIELDS
        saved_rows = list(saved_reader)
    assert len(saved_rows) == 1
    assert saved_rows[0]["anneal_head"] == "rev-measured-111"
    assert saved_rows[0]["code_revision"] == "rev-measured-111"
    assert saved_rows[0]["host"] == "bench"
    assert saved_rows[0]["machine_id"] == measured_machine["machine_id"]
    assert saved_rows[0]["cpu_model"] == measured_machine["cpu_model"]
    assert summary_measured[0]["anneal_head"] == "rev-measured-111"
    assert summary_measured[0]["code_revision"] == "rev-measured-111"
    assert summary_measured[0]["host"] == "bench"
    assert summary_measured[0]["machine_id"] == measured_machine["machine_id"]
    assert summary_measured[0]["cpu_model"] == measured_machine["cpu_model"]

    metal.write_saved_row(
        saved_seed0,
        saved_rows[0],
        anneal_head="rev-measured-111",
        code_revision="rev-measured-111",
        host="host-measured",
        machine=measured_machine,
    )
    reused_summary = fake_campaign_run(
        work_measure,
        head="rev-current-999",
        machine={"machine_id": "9" * 64, "cpu_model": "Current CPU"},
        seeds=(0,),
        summary_path=work_measure / "summary-reused.csv",
    )
    assert reused_summary[0]["anneal_head"] == "rev-measured-111"
    assert reused_summary[0]["code_revision"] == "rev-measured-111"
    assert reused_summary[0]["host"] == "host-measured"
    assert reused_summary[0]["machine_id"] == measured_machine["machine_id"]
    assert reused_summary[0]["cpu_model"] == measured_machine["cpu_model"]

    work_legacy = tmp_path / "legacy"
    legacy_runs = work_legacy / "runs" / "zn38"
    legacy_runs.mkdir(parents=True)
    (legacy_runs / "anneal-0.row").write_text(
        "zn38,anneal,0,20000,1,400,400,0.200,-10.0,0.150,0.050,3e-4,1e-4\n"
    )
    (legacy_runs / "anneal-1.row").write_text(
        ",".join(metal.CORE_ROW_FIELDS)
        + "\nzn38,anneal,1,20000,1,600,600,0.300,-10.0,0.250,0.050,4e-4,1e-4\n"
    )
    legacy_summary = fake_campaign_run(
        work_legacy,
        head="rev-current-999",
        machine={"machine_id": "9" * 64, "cpu_model": "Current CPU"},
        seeds=(0, 1),
        summary_path=work_legacy / "summary-legacy.csv",
    )
    assert len(legacy_summary) == 1
    for field in ("anneal_head", "code_revision", "host", "machine_id", "cpu_model"):
        assert legacy_summary[0][field] == "unknown"

    mixed_summary = fake_campaign_run(
        work_legacy,
        head="rev-fresh-222",
        machine={"machine_id": "2" * 64, "cpu_model": "Fresh CPU"},
        seeds=(0, 2),
        summary_path=work_legacy / "summary-mixed.csv",
    )
    assert mixed_summary[0]["anneal_head"] == "rev-fresh-222;unknown"
    assert mixed_summary[0]["code_revision"] == "rev-fresh-222;unknown"
    assert mixed_summary[0]["host"] == "bench;unknown"
    assert mixed_summary[0]["machine_id"] == f"{'2' * 64};unknown"
    assert mixed_summary[0]["cpu_model"] == "Fresh CPU;unknown"
