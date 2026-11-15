"""Checkpoint provenance the UMA exporter embeds in a ``.pt2``.

The exporter resolves the checkpoint once (``--checkpoint``,
``RGPOT_UMA_CHECKPOINT`` or a Hugging Face download) and records where
it came from and the SHA-256 of the file it loaded. Hugging Face is
replaced by a fake snapshot directory, so nothing here needs the network.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ase")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_uma_aoti as exp  # noqa: E402

PROVENANCE_KEYS = (
    "checkpoint_source",
    "checkpoint_path",
    "checkpoint_hf_repo",
    "checkpoint_hf_filename",
    "checkpoint_hf_revision",
    "checkpoint_sha256",
)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
WEIGHTS = b"not a real UMA checkpoint, only bytes to hash\n"


class _Double(torch.nn.Module):
    def forward(self, pos):
        return pos.sum(), 2.0 * pos


def _package(path: Path, meta: dict) -> Path:
    example = (torch.arange(6, dtype=torch.float64).reshape(2, 3),)
    program = torch.export.export(_Double(), example, strict=False)
    return exp.aoti_package(program, path, metadata=meta)


def _metadata(checkpoint=None) -> dict:
    example = [torch.zeros(3, 3, dtype=torch.float64)]
    return exp.runtime_metadata(
        Path("tiny.pt2"),
        6.0,
        300,
        example,
        torch.float64,
        model_dtype=torch.float64,
        label="tiny",
        model="uma-s-1p1",
        checkpoint=checkpoint,
    )


def _fake_hf(tmp_path, monkeypatch):
    """A Hugging Face cache snapshot holding ``WEIGHTS`` at ``COMMIT``."""
    snapshot = tmp_path / "models--facebook--UMA" / "snapshots" / COMMIT
    weights = snapshot / "checkpoints" / "uma-s-1p1.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(WEIGHTS)
    calls = []

    def registry(model):
        assert model == "uma-s-1p1"
        return "facebook/UMA", "uma-s-1p1.pt", "checkpoints", None

    def download(repo_id, filename, subfolder, revision):
        calls.append((repo_id, filename, subfolder, revision))
        return str(weights)

    monkeypatch.setattr(exp, "_hf_registry_entry", registry)
    monkeypatch.setattr(exp, "_hf_download", download)
    return weights, calls


def test_cli_checkpoint_wins_over_the_environment(tmp_path):
    cli = tmp_path / "cli.pt"
    env = tmp_path / "env.pt"
    cli.write_bytes(WEIGHTS)
    env.write_bytes(b"other weights")
    source = exp.resolve_checkpoint(
        str(cli), "uma-s-1p1", environ={"RGPOT_UMA_CHECKPOINT": str(env)}
    )
    assert source.kind == "cli"
    assert source.path == cli.resolve()
    assert source.sha256 == hashlib.sha256(WEIGHTS).hexdigest()


def test_environment_checkpoint_is_recorded_as_env(tmp_path):
    env = tmp_path / "env.pt"
    env.write_bytes(WEIGHTS)
    source = exp.resolve_checkpoint(
        None, "uma-s-1p1", environ={"RGPOT_UMA_CHECKPOINT": str(env)}
    )
    meta = source.metadata()
    assert meta == {
        "checkpoint_source": "env",
        "checkpoint_path": str(env.resolve()),
        "checkpoint_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
    }


def test_missing_local_checkpoint_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        exp.resolve_checkpoint(str(tmp_path / "absent.pt"), "uma-s-1p1", environ={})


def test_hf_download_records_repo_file_and_resolved_commit(tmp_path, monkeypatch):
    weights, calls = _fake_hf(tmp_path, monkeypatch)
    source = exp.resolve_checkpoint(None, "uma-s-1p1", environ={})
    assert calls == [("facebook/UMA", "uma-s-1p1.pt", "checkpoints", None)]
    assert source.kind == "hf"
    assert source.path == weights
    assert source.metadata() == {
        "checkpoint_source": "hf",
        "checkpoint_hf_repo": "facebook/UMA",
        "checkpoint_hf_filename": "checkpoints/uma-s-1p1.pt",
        "checkpoint_hf_revision": COMMIT,
        "checkpoint_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
    }


def test_hf_download_outside_a_snapshot_is_refused(tmp_path, monkeypatch):
    loose = tmp_path / "uma-s-1p1.pt"
    loose.write_bytes(WEIGHTS)
    monkeypatch.setattr(
        exp, "_hf_registry_entry",
        lambda model: ("facebook/UMA", "uma-s-1p1.pt", "checkpoints", "main"),
    )
    monkeypatch.setattr(exp, "_hf_download", lambda *args: str(loose))
    with pytest.raises(SystemExit, match="commit"):
        exp.resolve_checkpoint(None, "uma-s-1p1", environ={})


def test_every_load_reads_the_file_that_was_hashed(tmp_path, monkeypatch):
    """The predict unit loads the resolved file, never a second download."""
    pytest.importorskip("fairchem.core")
    import fairchem.core.units.mlip_unit as mlip_unit

    weights, calls = _fake_hf(tmp_path, monkeypatch)
    source = exp.resolve_checkpoint(None, "uma-s-1p1", environ={})
    loaded = []

    def fake_load(path, *args, **kwargs):
        loaded.append(Path(path))
        return "unit"

    monkeypatch.setattr(mlip_unit, "load_predict_unit", fake_load)
    monkeypatch.setattr(exp, "_hf_references", lambda model: (None, None))
    assert exp.predict_unit("uma-s-1p1", "cpu", None, source) == "unit"
    assert exp.predict_unit("uma-s-1p1", "cpu", None, source) == "unit"
    assert loaded == [weights, weights]
    assert len(calls) == 1


def test_cli_package_names_the_digest_of_its_weights(tmp_path):
    ckpt = tmp_path / "uma-s-1p1.pt"
    ckpt.write_bytes(WEIGHTS)
    source = exp.resolve_checkpoint(str(ckpt), "uma-s-1p1", environ={})
    out = _package(tmp_path / "cli.pt2", _metadata(source))
    meta = exp.package_metadata(out)
    assert meta["checkpoint_source"] == "cli"
    assert meta["checkpoint_path"] == str(ckpt.resolve())
    assert meta["checkpoint_sha256"] == hashlib.sha256(WEIGHTS).hexdigest()
    assert "checkpoint_hf_revision" not in meta
    # torch's own loader reads the same keys UmaPot reads.
    assert exp.load_aoti(out).get_metadata()["checkpoint_sha256"] == (
        hashlib.sha256(WEIGHTS).hexdigest()
    )


def test_hf_package_names_the_resolved_commit(tmp_path, monkeypatch):
    _fake_hf(tmp_path, monkeypatch)
    source = exp.resolve_checkpoint(None, "uma-s-1p1", environ={})
    out = _package(tmp_path / "hf.pt2", _metadata(source))
    prov = exp.checkpoint_provenance(exp.package_metadata(out))
    assert prov == {
        "checkpoint_source": "hf",
        "checkpoint_path": None,
        "checkpoint_hf_repo": "facebook/UMA",
        "checkpoint_hf_filename": "checkpoints/uma-s-1p1.pt",
        "checkpoint_hf_revision": COMMIT,
        "checkpoint_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
    }


def test_package_without_provenance_loads_and_reports_it_absent(tmp_path):
    out = _package(tmp_path / "old.pt2", _metadata())
    meta = exp.load_aoti(out).get_metadata()
    for key in PROVENANCE_KEYS:
        assert key not in meta
    prov = exp.checkpoint_provenance(meta)
    assert prov == {key: None for key in PROVENANCE_KEYS}
    assert exp.checkpoint_provenance(exp.package_metadata(out)) == prov
    pos = torch.ones(2, 3, dtype=torch.float64)
    energy, forces = exp.run_aoti(exp.load_aoti(out), (pos,))
    torch.testing.assert_close(energy, pos.sum())
    torch.testing.assert_close(forces, 2.0 * pos)
