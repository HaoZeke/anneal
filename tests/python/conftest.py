"""Pytest hooks for the free-threaded ASE suite.

Tests import the installed rgpot. The source tree under python/ is not
placed on sys.path, because that would shadow the wheel.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, NoReturn

import pytest

# Local helpers such as support.py. This directory is not the package.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

_TRUTHY = {"1", "true", "yes", "on"}


class PotservServer:
    """Running potserv subprocess bound to a localhost TCP port."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        potential: str,
        binary: Path,
        process: subprocess.Popen[str],
    ) -> None:
        self.host = host
        self.port = port
        self.potential = potential
        self.binary = binary
        self.process = process


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _rpc_required() -> bool:
    return _is_truthy(os.environ.get("CI")) or _is_truthy(
        os.environ.get("REQUIRE_RPC")
    )


def _find_potserv_binary() -> tuple[Path | None, str | None]:
    for env_name in ("RGPOT_POTSERV_BIN", "RGPOT_POTSERV"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            candidate = Path(raw)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate, None
            return None, f"{env_name}={raw!r} is not an executable potserv binary"

    names = ("potserv.exe", "potserv") if sys.platform == "win32" else ("potserv",)
    for rel_dir in (("bbdir", "CppCore"), ("build", "CppCore"), ("build",)):
        for name in names:
            candidate = _REPO_ROOT.joinpath(*rel_dir, name)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate, None

    which = shutil.which("potserv")
    if which:
        candidate = Path(which)
        if candidate.is_file():
            return candidate, None

    return None, None


def _missing_rpc(reason: str) -> NoReturn:
    msg = (
        f"{reason}. Build potserv with -Dwith_rpc=true or set "
        "RGPOT_POTSERV_BIN=/path/to/potserv (or RGPOT_POTSERV); "
        "select RPC tests with `-m rpc`, set REQUIRE_RPC=1 to fail when "
        "unavailable, or pass `-m 'not rpc'` to deselect them."
    )
    if _rpc_required():
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg)


def _free_port(host: str = "localhost") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_for_port(
    host: str,
    port: int,
    proc: subprocess.Popen[str],
    *,
    potential: str,
    timeout: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout
    last_err: OSError | None = None
    while True:
        rc = proc.poll()
        if rc is not None:
            stdout, stderr = proc.communicate()
            detail = (stderr or stdout or "").strip()
            raise RuntimeError(
                f"potserv exited prematurely (code {rc}) for potential "
                f"{potential!r}: {detail}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_err = exc
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"potserv did not listen on {host}:{port} within "
                    f"{timeout}s (last error: {last_err})"
                ) from exc
            time.sleep(0.05)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "thread_unsafe: not safe under pytest-run-parallel",
    )
    config.addinivalue_line(
        "markers",
        "rpc: requires the potserv RPC server binary and pycapnp",
    )
    # Import after this directory is on sys.path. Test modules then see the
    # rootdir pytest chose. A checkout used as that rootdir is the source tree.
    import repo_data

    repo_data.PYTEST_ROOT = Path(config.rootpath).resolve()


@pytest.fixture
def potserv_bin() -> Path:
    """Resolve the potserv executable, skipping locally or failing in CI."""
    binary, detail = _find_potserv_binary()
    if binary is None:
        _missing_rpc(detail or "potserv server binary was not found")
    try:
        import capnp  # noqa: F401
    except ImportError as exc:
        _missing_rpc(f"pycapnp is not installed ({exc})")
    return binary


@pytest.fixture
def potserv_factory(
    potserv_bin: Path,
) -> Iterator[Callable[..., PotservServer]]:
    """Start potserv on a free localhost port, wait for readiness, and tear down."""
    started: list[subprocess.Popen[str]] = []

    def _start(
        potential: str = "CuH2",
        *,
        startup_timeout: float = 15.0,
    ) -> PotservServer:
        host = "localhost"
        port = _free_port(host)
        proc: subprocess.Popen[str] = subprocess.Popen(
            [str(potserv_bin), str(port), potential],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started.append(proc)
        try:
            _wait_for_port(
                host,
                port,
                proc,
                potential=potential,
                timeout=startup_timeout,
            )
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
            raise
        return PotservServer(
            host=host,
            port=port,
            potential=potential,
            binary=potserv_bin,
            process=proc,
        )

    try:
        yield _start
    finally:
        for proc in reversed(started):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()


@pytest.fixture
def potserv_server(
    potserv_factory: Callable[..., PotservServer],
) -> PotservServer:
    """Start a live potserv server with a built-in potential."""
    import rgpot

    potential = "CuH2" if rgpot.has_fortran_pots else "LJ"
    return potserv_factory(potential)
