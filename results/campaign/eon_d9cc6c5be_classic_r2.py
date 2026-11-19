"""Resumable eon-classic rerun for the Pt-island saddle catalog.

The search loop, seed, goal, and template overrides match
``rgsaddle_develop_catalog.py``. This script writes the versioned r2
files and leaves the existing campaign files untouched. A work tree
that already contains ``states/`` continues that run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = "eon-d9cc6c5be-saddle-pt-classic-r2"
CAMPAIGN = ROOT / "results" / "campaign"
DAT_PATH = CAMPAIGN / f"{STEM}.dat"
STARTS_PATH = CAMPAIGN / f"{STEM}.starts.jsonl"
V2_PATH = CAMPAIGN / f"{STEM}.v2.json"
PROVENANCE_PATH = CAMPAIGN / f"{STEM}-eon-provenance.json"
CHECKPOINT = ROOT / "build" / "eon-d9cc6c5be-classic-r2"
REFERENCE_HARDWARE = CAMPAIGN / "rgsaddle-develop-179e90f2-pt-island-hardware.json"
EON_COMMIT = "d9cc6c5bee432d625eb6c56d7ea74b723a37ee3b"
RGSADDLE_COMMIT = "aa6dd9371d6a63103010a6829cb62c74a39c7ff1"
TEMPLATE_ID = "eon/develop-saddle-pt-dimer"
THREAD_VARS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class Labels:
    """Replace absolute paths with stable labels and hash each realpath."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._roots: list[tuple[str, str]] = []

    def add(self, label: str, path: Path, *, binary: bool = False) -> None:
        real = path.resolve()
        item = {
            "label": label,
            "realpath_sha256": _sha256_bytes(str(real).encode()),
        }
        if binary and real.is_file():
            item["binary_sha256"] = _sha256_file(real)
        self.entries.append(item)
        self._roots.append((str(real), label))
        self._roots.sort(key=lambda pair: len(pair[0]), reverse=True)

    def rewrite(self, value: str) -> str:
        text = value
        for root, label in self._roots:
            text = text.replace(root, label)
        return text

    def rewrite_obj(self, value):
        if isinstance(value, str):
            return self.rewrite(value)
        if isinstance(value, list):
            return [self.rewrite_obj(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.rewrite_obj(item) for key, item in value.items()}
        return value


class LoggingPopen(subprocess.Popen):
    """Record each child process the communicator starts."""

    log_path: Path | None = None

    def __init__(self, args, *pos, **kwargs):
        argv = list(args) if isinstance(args, (list, tuple)) else [str(args)]
        self._optbench_done = False
        self._optbench_t0 = time.perf_counter()
        self._optbench_meta = {
            "argv": [str(item) for item in argv],
            "cwd": str(kwargs.get("cwd") or ""),
        }
        super().__init__(args, *pos, **kwargs)

    def _optbench_finish(self) -> None:
        if self._optbench_done or self.returncode is None or self.log_path is None:
            return
        self._optbench_done = True
        record = dict(self._optbench_meta)
        record["returncode"] = self.returncode
        record["wall_seconds"] = time.perf_counter() - self._optbench_t0
        _append_jsonl(self.log_path, record)

    def poll(self, timeout=None):
        code = super().poll()
        self._optbench_finish()
        return code

    def wait(self, timeout=None):
        code = super().wait(timeout)
        self._optbench_finish()
        return code


def _install_popen_log(path: Path) -> None:
    LoggingPopen.log_path = path
    subprocess.Popen = LoggingPopen


def _pin_process(cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    for name in THREAD_VARS:
        os.environ[name] = "1"
    os.environ["LJOS_SEAT"] = "benchbot"


def _affinity() -> str:
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))


def _first_line(argv: list[str]) -> str:
    out = subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)
    return out.splitlines()[0].strip()


def _client_version(client: Path) -> str:
    out = subprocess.check_output([str(client), "--version"], text=True, stderr=subprocess.STDOUT)
    return " ".join(out.split())


def _lscpu() -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        text = subprocess.check_output(["lscpu"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return found
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        found[key.strip()] = value.strip()
    return found


def _mem_total_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _machine_id_sha256() -> str:
    try:
        raw = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    return _sha256_bytes(raw.encode())


def _hardware() -> dict:
    cpu = _lscpu()
    return {
        "model_name": cpu.get("Model name", ""),
        "vendor_id": cpu.get("Vendor ID", ""),
        "cpu_family": cpu.get("CPU family", ""),
        "model": cpu.get("Model", ""),
        "stepping": cpu.get("Stepping", ""),
        "cpu_cores": cpu.get("CPU(s)", ""),
        "cores_per_socket": cpu.get("Core(s) per socket", ""),
        "sockets": cpu.get("Socket(s)", ""),
        "threads_per_core": cpu.get("Thread(s) per core", ""),
        "cpus_online": cpu.get("On-line CPU(s) list", ""),
        "l1d_cache": cpu.get("L1d cache", ""),
        "l1i_cache": cpu.get("L1i cache", ""),
        "l2_cache": cpu.get("L2 cache", ""),
        "l3_cache": cpu.get("L3 cache", ""),
        "mem_total_kb": _mem_total_kb(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "machine_id_sha256": _machine_id_sha256(),
    }


def _hardware_comparison(measured: dict) -> dict:
    recorded = _read_json(REFERENCE_HARDWARE)
    pairs = {
        "model_name": (str(recorded.get("model_name", "")), measured.get("model_name", "")),
        "cpu_cores": (str(recorded.get("cpu_cores", "")), str(measured.get("cpu_cores", ""))),
        "cpus_online": (str(recorded.get("cpus_online", "")), str(measured.get("cpus_online", ""))),
        "mem_total_kb": (str(recorded.get("mem_total_kb", "")), str(measured.get("mem_total_kb", ""))),
    }
    fields = {}
    for key, (previous, current) in pairs.items():
        fields[key] = {
            "recorded": previous,
            "this_run": current,
            "match": previous == current and previous != "",
        }
    matched = [key for key, item in fields.items() if item["match"]]
    differed = [key for key, item in fields.items() if item["recorded"] and not item["match"]]
    return {
        "reference": "results/campaign/rgsaddle-develop-179e90f2-pt-island-hardware.json",
        "fields": fields,
        "matched_fields": matched,
        "differed_fields": differed,
        "unrecorded_on_other_arms": [
            "cpu_family",
            "model",
            "stepping",
            "l1d_cache",
            "l1i_cache",
            "l2_cache",
            "l3_cache",
            "kernel_release",
        ],
        "same_hardware_claimed": False,
        "note": (
            "The other arms of this catalog store model name, core count, "
            "online cpu list, and memory. Family, stepping, caches, and kernel "
            "were not stored there, so this file does not claim the machines are the same."
            if not differed
            else "Recorded hardware fields differ from this machine: " + ", ".join(differed) + "."
        ),
    }


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


def _compiler_block(labels: Labels, build: dict) -> dict:
    recorded = (build.get("compilers") or {}) if build else {}
    out = {}
    for key, fallback in (("cc", "gcc"), ("cxx", "g++"), ("fc", "gfortran")):
        item = recorded.get(key) or {}
        path = Path(item["path"]) if item.get("path") else _which(fallback)
        if path is None:
            continue
        label = f"<{key}>"
        labels.add(label, path, binary=True)
        version = item.get("version_first_line") or _first_line([str(path), "--version"])
        out[key] = {
            "label": label,
            "version_first_line": version,
            "binary_sha256": _sha256_file(path.resolve()),
            "realpath_sha256": _sha256_bytes(str(path.resolve()).encode()),
        }
    return out


def _tool_block(labels: Labels, build: dict, key: str, fallback: str) -> dict:
    item = (build.get(key) or {}) if build else {}
    path = Path(item["path"]) if item.get("path") else _which(fallback)
    if path is None:
        return {"version": item.get("version", "")}
    label = f"<{key}>"
    labels.add(label, path, binary=True)
    version = item.get("version") or ""
    if not version:
        try:
            version = subprocess.check_output([str(path), "--version"], text=True).splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError):
            version = ""
    return {
        "label": label,
        "version": version,
        "binary_sha256": _sha256_file(path.resolve()) if path.resolve().is_file() else "",
        "realpath_sha256": _sha256_bytes(str(path.resolve()).encode()),
    }


def _pair_walls(records: list[dict], invocations: list[dict]) -> list[dict]:
    clients = [
        item
        for item in invocations
        if Path(str(item.get("argv", [""])[-1])).name == "eonclient"
    ]
    paired = [dict(record) for record in records]
    offset = 0
    if len(clients) == len(records) + 1:
        offset = 1
    if len(clients) - offset == len(records):
        for record, invocation in zip(paired, clients[offset:]):
            record["wall_seconds"] = invocation.get("wall_seconds")
    return paired


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _write_dat(records: list[dict], wall: float, version: str, error: str, catalog) -> None:
    connected = sum(1 for record in records if record["success"])
    calls = [int(record["force_calls"]) for record in records]
    stopped = connected >= catalog.GOAL or len(records) >= catalog.MAX_SEARCHES
    lines = [
        f"# rep 1 force_calls {sum(calls)} total_searches {len(records)} connected {connected} seed {catalog.SEED}",
        "code eOn",
        f"code_version {version}",
        f"commit {EON_COMMIT}",
        "algorithm Dimer",
        "dimer_improved false",
        "template eon/develop-saddle-pt-dimer",
        "problem saddle-search/pt-island",
        "counter client-search-results",
        "count_keys sad-fcs+mins-fcs",
        f"seed {catalog.SEED}",
        "communicator_jobs 1",
        "repetitions 1",
        f"total_searches {len(records)}",
        f"connected {connected}",
        f"goal {catalog.GOAL}",
        f"nfailed {0 if stopped else 1}",
        f"wall_seconds {wall:.6f}",
        "cpu_affinity 0",
    ]
    if error:
        lines.append("error " + error.splitlines()[0])
    DAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DAT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with STARTS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_v2(records: list[dict], version: str, cap: int | None) -> None:
    from optbench.score import score_v2

    metrics = score_v2(
        [
            {
                "force_calls": int(record["force_calls"]),
                "success": bool(record["success"]),
                "reasons": record.get("reasons") or [],
                "result": record.get("result"),
            }
            for record in records
        ],
        cap,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": "saddle-search/pt-island",
        "scoring": "par2-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "One row is one displacement search from the eon-classic rerun. "
            "The cap is the template failure_cap used for this dimer template."
        ),
        "runs": [
            {
                "entry": STEM,
                "code": "eOn",
                "algorithm": "Dimer",
                "code_version": version,
                "commit": EON_COMMIT,
                "present_source": True,
                "cap": cap,
                "provenance": {
                    "code": "eOn",
                    "code_version": version,
                    "commit": EON_COMMIT,
                    "machine": {
                        "machine": platform.machine(),
                        "node": "unspecified",
                        "python": platform.python_version(),
                        "system": platform.system(),
                    },
                    "parameters": {
                        "cap": cap,
                        "problem": "saddle-search/pt-island",
                        "template": TEMPLATE_ID,
                    },
                },
                "metrics": metrics,
                "starts": records,
            }
        ],
    }
    V2_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _provenance(
    *,
    client: Path,
    version: str,
    sha_before: str,
    sha_after: str,
    build: dict,
    work: Path,
    config_text: str,
    records: list[dict],
    invocations: list[dict],
    segments: list[dict],
    resumed: bool,
    wall: float,
    error: str,
    catalog_budget: str,
    template_cap: int | None,
) -> None:
    labels = Labels()
    prefix = client.resolve().parent.parent
    labels.add("<eon-prefix>", prefix)
    labels.add("<work-prefix>", work)
    labels.add("<repo>", ROOT)
    source = build.get("source")
    builddir = build.get("builddir")
    if source:
        labels.add("<eon-src>", Path(source))
    if builddir:
        labels.add("<eon-build>", Path(builddir))
    compilers = _compiler_block(labels, build)
    meson = _tool_block(labels, build, "meson", "meson")
    ninja = _tool_block(labels, build, "ninja", "ninja")
    rustc = _tool_block(labels, build, "rustc", "rustc")
    commands = labels.rewrite_obj(list(build.get("commands") or []))
    client_invocations = []
    for index, item in enumerate(invocations, start=1):
        argv = [str(part) for part in item.get("argv") or []]
        if not argv or Path(argv[-1]).name != "eonclient":
            continue
        client_invocations.append(
            {
                "index": index,
                "argv": [labels.rewrite(part) for part in argv],
                "cwd": labels.rewrite(str(item.get("cwd") or "")),
                "returncode": item.get("returncode"),
                "wall_seconds": item.get("wall_seconds"),
                "taskset": "0",
            }
        )
    calls = [float(record["force_calls"]) for record in records]
    connected_calls = [float(record["force_calls"]) for record in records if record["success"]]
    walls = [float(record["wall_seconds"]) for record in records if record.get("wall_seconds") is not None]
    connected = sum(1 for record in records if record["success"])
    measured = _hardware()
    payload = {
        "arm": "eon-classic",
        "version_suffix": "r2",
        "base": "9cd4ea7b3e7c11591bede7aabb1e3c1a5406f0e8",
        "spec": {
            "runner": "results/campaign/rgsaddle_develop_catalog.py",
            "template": TEMPLATE_ID,
            "seed": 20261016,
            "goal_connected": 27,
            "max_searches": 20000,
            "prior_files": [
                "results/campaign/eon-3.5.0-d9cc6c5be-saddle-pt-classic-aa6dd937.dat",
                "results/campaign/eon-3.5.0-d9cc6c5be-saddle-pt-classic-aa6dd937.starts.jsonl",
                "results/campaign/rgsaddle-develop-aa6dd937-eon-provenance.json",
            ],
            "prior_v2": "The aa6dd937 eon-classic arm has no v2 file.",
            "prior_ref": (
                "No optbench branch or pull request at this fetch contains the aa6dd937 campaign. "
                "The protocol files are the eon-classic products of the patch titled "
                "feat: record the aa6dd937 pt-island catalog."
            ),
        },
        "prefixes": labels.entries,
        "eon": {
            "repository": "https://github.com/TheochemUI/eOn",
            "commit": EON_COMMIT,
            "tag": "v3.5.0",
        },
        "eonclient": {
            "path": "<eon-prefix>/bin/eonclient",
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "sha256_match": sha_before == sha_after,
            "version_stdout": version,
        },
        "build": {
            "commands": commands,
            "compilers": compilers,
            "meson": meson,
            "ninja": ninja,
            "rustc": rustc,
            "prefix": "<eon-prefix>",
            "source": "<eon-src>" if source else "",
            "builddir": "<eon-build>" if builddir else "",
        },
        "rgsaddle": {
            "repository": "https://github.com/OmniPotentRPC/rgsaddle",
            "develop_commit": RGSADDLE_COMMIT,
            "role": (
                "The eon-classic arm runs eonclient with potential morse_pt. "
                "librgsaddle is unused, so this rerun does not build it."
            ),
        },
        "run": {
            "argv": [
                "taskset",
                "-c",
                "0",
                "python3",
                "results/campaign/eon_d9cc6c5be_classic_r2.py",
                "--client",
                "<eon-prefix>/bin/eonclient",
            ],
            "cpu_affinity": _affinity(),
            "communicator_jobs": 1,
            "number_of_cpus": 1,
            "threads": {name: os.environ.get(name, "") for name in THREAD_VARS},
            "env": {
                "LJOS_SEAT": os.environ.get("LJOS_SEAT", ""),
                **{name: os.environ.get(name, "") for name in THREAD_VARS},
            },
            "seed": 20261016,
            "resumed": resumed,
            "segments": segments,
            "wall_seconds": wall,
            "error": error.splitlines()[0] if error else "",
            "config_ini": labels.rewrite(config_text),
            "client_commands": client_invocations,
        },
        "budgets": {
            "goal_connected": 27,
            "max_searches": 20000,
            "template_failure_cap": template_cap,
            "catalog_force_call_budget": catalog_budget,
        },
        "summary": {
            "searches": len(records),
            "connected": connected,
            "force_call_median": _median(calls),
            "force_call_median_connected": _median(connected_calls),
            "wall_median": _median(walls),
            "wall_seconds": wall,
        },
        "hardware": measured,
        "hardware_comparison": _hardware_comparison(measured),
    }
    PROVENANCE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_segments() -> list[dict]:
    path = CHECKPOINT / "segments.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _save_segments(segments: list[dict]) -> None:
    path = CHECKPOINT / "segments.json"
    path.write_text(json.dumps(segments, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _segment_wall(segments: list[dict]) -> float:
    total = 0.0
    for segment in segments:
        if segment.get("wall_seconds") is not None:
            total += float(segment["wall_seconds"])
    return total


def run(client: Path, build_record: Path | None, cpu: int) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _pin_process(cpu)
    _install_popen_log(CHECKPOINT / "invocations.jsonl")
    catalog = _load_catalog()
    from optbench.campaign import load_template
    from optbench.eon_akmc import _write_config, prepare_client_env
    from optbench.eon_parse import saddle_progress

    version = _client_version(client)
    sha_before = _sha256_file(client.resolve())
    folder = CHECKPOINT
    folder.mkdir(parents=True, exist_ok=True)
    work = folder / "work"
    resumed = (work / "states").is_dir()
    template = copy.deepcopy(load_template(ROOT, TEMPLATE_ID))
    template["ini"]["Communicator"]["num_jobs"] = 1
    template["ini"]["Communicator"]["number_of_CPUs"] = 1
    template["ini"]["Main"]["random_seed"] = catalog.SEED
    if not resumed:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        pos = ROOT / "website" / "saddle-search" / "pt-island" / "pt-island.con"
        if not pos.is_file():
            from optbench.campaign import geometry_archive

            pos = geometry_archive(
                ROOT,
                {"id": "saddle", "input": {"archive_path": "website/saddle-search/pt-island/pt-island.con"}},
            )
        shutil.copy(pos, work / "pos.con")
        _write_config(template, work / "config.ini", client, catalog.SEED, 1)
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (work / relative).mkdir(parents=True, exist_ok=True)
    prepare_client_env(client)
    import logging

    for name in ("akmc", "eon", "state", "communicator"):
        logging.getLogger(name).setLevel(logging.WARNING)
    segments = _load_segments()
    if segments and "wall_seconds" not in segments[-1]:
        segments[-1]["interrupted"] = True
    segment = {
        "started_unix": time.time(),
        "resumed": resumed,
        "cpu_affinity": _affinity(),
    }
    segments.append(segment)
    _save_segments(segments)
    started = time.perf_counter()
    previous = Path.cwd()
    last_error = ""
    records: list[dict] = []
    try:
        os.chdir(work)
        import inspect

        from eon import akmc as akmc_mod
        from eon.config import ConfigClass
        from eon.config import config as shared_config

        parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
        if "config" in parameters:
            config = ConfigClass()
        else:
            config = shared_config
            config.init_done = False
        config.init()
        from eon import communicator

        if hasattr(communicator.get_communicator, "comm"):
            del communicator.get_communicator.comm
        for index in range(catalog.MAX_SEARCHES):
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                break
            results = work / "states" / "0" / "search_results.txt"
            if not results.is_file():
                continue
            text = results.read_text(encoding="utf-8", errors="replace")
            records = catalog._eon_records(text)
            _write_ledger(folder / "searches.jsonl", records)
            progress = saddle_progress(text)
            connected = sum(1 for record in records if record["success"])
            if (index + 1) % 25 == 0 or connected >= catalog.GOAL or len(records) >= catalog.MAX_SEARCHES:
                print(
                    f"eon-classic-r2 searches {len(records)} connected {connected} "
                    f"qualifying {progress['qualifying']}",
                    flush=True,
                )
            if connected >= catalog.GOAL or len(records) >= catalog.MAX_SEARCHES or progress["reached"]:
                break
    finally:
        os.chdir(previous)
    results = work / "states" / "0" / "search_results.txt"
    if results.is_file():
        records = catalog._eon_records(results.read_text(encoding="utf-8", errors="replace"))
        _write_ledger(folder / "searches.jsonl", records)
    duration = time.perf_counter() - started
    segment["finished_unix"] = time.time()
    segment["wall_seconds"] = duration
    _save_segments(segments)
    wall = _segment_wall(segments)
    invocations = _load_jsonl(folder / "invocations.jsonl")
    paired = _pair_walls(records, invocations)
    sha_after = _sha256_file(client.resolve())
    _write_dat(paired, wall, version, last_error, catalog)
    from optbench.board import _cap_of
    from optbench.campaign import load_template, resolve_problem

    loaded = load_template(ROOT, TEMPLATE_ID)
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    template_cap = _cap_of(problem, loaded, {})
    if paired:
        _write_v2(paired, version, template_cap)
    budget = catalog._protocol_budget("eon-classic", {"template": TEMPLATE_ID})
    config_text = (work / "config.ini").read_text(encoding="utf-8") if (work / "config.ini").is_file() else ""
    build = _read_json(build_record) if build_record else {}
    _provenance(
        client=client,
        version=version,
        sha_before=sha_before,
        sha_after=sha_after,
        build=build,
        work=work,
        config_text=config_text,
        records=paired,
        invocations=invocations,
        segments=segments,
        resumed=resumed or any(bool(item.get("resumed")) for item in segments) or len(segments) > 1,
        wall=wall,
        error=last_error,
        catalog_budget=str(budget),
        template_cap=template_cap,
    )
    if sha_before != sha_after:
        raise SystemExit("eonclient sha256 changed during the run")
    if last_error:
        raise SystemExit(last_error.splitlines()[0])


def _write_ledger(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _load_catalog():
    import importlib.util

    path = CAMPAIGN / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_taskset(cpu: int) -> None:
    if os.sched_getaffinity(0) == {cpu}:
        return
    os.execvp("taskset", ["taskset", "-c", str(cpu), sys.executable, *sys.argv])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Rerun the eon-classic Pt-island arm")
    parser.add_argument("--client", required=True, type=Path)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args(argv)
    _ensure_taskset(args.cpu)
    run(args.client, args.build_record, args.cpu)


if __name__ == "__main__":
    main(sys.argv[1:])
