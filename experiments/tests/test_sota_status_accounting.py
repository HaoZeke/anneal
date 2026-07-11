import csv

import numpy as np

from experiments.scripts import sota_cutest
from experiments.scripts.summarize_sota import summarize_rows


def test_unexpected_method_error_is_not_scored_as_partial_success():
    counter = sota_cutest.Counter(lambda x: float(np.dot(x, x)), budget=8)

    def broken(counter, low, high, dim, grad, rng, anchor=None):
        del low, high, dim, grad, rng, anchor
        counter(np.array([0.5]))
        raise RuntimeError("solver failed")

    row = sota_cutest.run_method_cell(
        method_name="broken",
        method=broken,
        problem="sphere",
        dim=1,
        seed=3,
        counter=counter,
        low=np.array([-1.0]),
        high=np.array([1.0]),
        grad=None,
        anchor=np.array([0.0]),
    )

    assert row["status"] == "error:RuntimeError"
    assert row["best"] == float("inf")
    assert row["evals"] == 1
    assert row["objective_evals"] == 1
    assert row["grad_evals"] == 0


def test_summary_excludes_failed_rows_and_uses_average_tie_ranks():
    rows = [
        {"problem": "p", "seed": "0", "method": "a", "best": "1", "status": "ok"},
        {"problem": "p", "seed": "0", "method": "b", "best": "1", "status": "ok"},
        {"problem": "p", "seed": "0", "method": "c", "best": "2", "status": "ok"},
        {
            "problem": "p",
            "seed": "0",
            "method": "failed",
            "best": "0",
            "status": "error:RuntimeError",
        },
    ]

    summary = summarize_rows(rows)

    assert summary["a"]["mean_rank"] == 1.5
    assert summary["b"]["mean_rank"] == 1.5
    assert summary["c"]["mean_rank"] == 3.0
    assert summary["a"]["wins"] == 1
    assert summary["b"]["wins"] == 1
    assert "failed" not in summary


def test_sota_csv_schema_keeps_status_and_split_work_counts(tmp_path):
    path = tmp_path / "row.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sota_cutest.FIELDNAMES)
        writer.writeheader()
    assert path.read_text().splitlines()[0].split(",") == sota_cutest.FIELDNAMES
    assert {"status", "objective_evals", "grad_evals", "evals"} <= set(
        sota_cutest.FIELDNAMES
    )


def test_problem_manifest_preserves_declared_order_kind_and_dimension(tmp_path):
    path = tmp_path / "problems.csv"
    path.write_text(
        "problem,kind,dim,stratum,selection_key\n"
        "ROSENBR,unconstrained,2,u_1_3,001\n"
        "BOX3,bound,3,b_1_3,002\n"
    )

    targets = sota_cutest.load_problem_manifest(path)

    assert [(t.name, t.kind, t.dim) for t in targets] == [
        ("ROSENBR", "unconstrained", 2),
        ("BOX3", "bound", 3),
    ]


def test_turbo_baseline_is_registered_and_updates_trust_region():
    assert sota_cutest.METHODS["turbo"] is sota_cutest.turbo
    state = sota_cutest._TurboState(dim=4, batch_size=1, best_value=1.0)
    state.failure_tolerance = 1
    updated = sota_cutest._turbo_update(state, np.array([[0.0]]))
    assert updated.length == 0.4
