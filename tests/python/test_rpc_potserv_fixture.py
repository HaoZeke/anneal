# Copyright 2023--present rgpot developers

"""Tests for the potserv pytest fixture skip/fail and lifecycle contract."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.thread_unsafe,
    pytest.mark.iterations(1),
]

_CONTRACT_TEST = (
    Path(__file__).resolve().parent
    / "test_ase_calculator_contract.py::test_rpc_calculator_contract_when_potserv_exists"
)


def _clean_rpc_env(potserv_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("CI", "REQUIRE_RPC", "RGPOT_POTSERV", "RGPOT_POTSERV_BIN"):
        env.pop(key, None)
    env["RGPOT_POTSERV_BIN"] = str(potserv_path)
    return env


def test_missing_potserv_skips_explicitly_locally(tmp_path: Path) -> None:
    env = _clean_rpc_env(tmp_path / "missing-potserv")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-rs", str(_CONTRACT_TEST)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "1 skipped" in combined
    assert "RGPOT_POTSERV_BIN" in combined
    assert "REQUIRE_RPC" in combined
    assert "rpc" in combined


@pytest.mark.parametrize(
    ("env_key", "env_val"),
    [("CI", "true"), ("REQUIRE_RPC", "1")],
)
def test_missing_potserv_fails_when_required(
    tmp_path: Path, env_key: str, env_val: str
) -> None:
    env = _clean_rpc_env(tmp_path / "missing-potserv")
    env[env_key] = env_val
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", str(_CONTRACT_TEST)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "SKIPPED" not in combined
    assert "1 skipped" not in combined
    assert "1 error" in combined or "1 failed" in combined
    assert "RGPOT_POTSERV_BIN" in combined


@pytest.mark.rpc
def test_potserv_factory_starts_waits_and_tears_down(tmp_path: Path) -> None:
    pid_file = tmp_path / "mock_potserv.pid"
    port_file = tmp_path / "mock_potserv.port"
    script = tmp_path / "mock_potserv.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import signal
            import socket
            import sys
            import time
            from pathlib import Path

            port = int(sys.argv[1])
            Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
            Path({str(port_file)!r}).write_text(str(port), encoding="utf-8")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", port))
            sock.listen(5)

            running = True

            def _stop(_signum, _frame):
                global running
                running = False

            signal.signal(signal.SIGTERM, _stop)
            sock.settimeout(0.1)
            while running:
                try:
                    conn, _ = sock.accept()
                    conn.close()
                except TimeoutError:
                    pass
            sock.close()
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)

    probe_test = tmp_path / "test_probe.py"
    probe_test.write_text(
        textwrap.dedent(
            """\
            import socket

            def test_fixture_lifecycle(potserv_factory):
                srv = potserv_factory("LJ")
                assert srv.port > 0
                assert srv.process.poll() is None
                with socket.create_connection((srv.host, srv.port), timeout=1.0):
                    pass
            """
        ),
        encoding="utf-8",
    )

    env = _clean_rpc_env(script)
    env["REQUIRE_RPC"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
            "-p",
            "conftest",
            "-o",
            f"pythonpath={Path(__file__).resolve().parent}",
            str(probe_test),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "1 passed" in combined
    assert pid_file.is_file()
    pid = int(pid_file.read_text(encoding="utf-8"))
    port = int(port_file.read_text(encoding="utf-8"))
    assert port > 0
    # After pytest exits, the fixture teardown must have stopped the server.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
