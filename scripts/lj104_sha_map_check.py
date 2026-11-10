#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the two sha-map check logs.

The stale log copies the map, writes the old prediction digests, and
runs the check. The current log restores that copy and runs the check
again. Neither log rewrites the restored map.
"""

import subprocess
import sys
from pathlib import Path

STALE = (
    "9b0bd7d3f94a720b5bd32c7fca66b62ccb1958aa8df5677f466d20e602a7fe61"
)
ROW = "scripts/lj104_v4_prediction.txt"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = ["git", "-c", f"safe.directory={repo.resolve()}", *args]
    return subprocess.run(
        cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def fail(what: str) -> None:
    print(what, file=sys.stderr)
    raise SystemExit(1)


def require_relative(token: str) -> None:
    if token.startswith("/"):
        fail("arguments must be relative")


def identity(kit_arg: str, repo_arg: str, repo: Path) -> list[str]:
    head = git(repo, "rev-parse", "HEAD")
    blob = git(repo, "rev-parse", "HEAD:scripts/lj104_sha_map.py")
    if head.returncode != 0 or blob.returncode != 0:
        fail("git rev-parse failed")
    return [
        f"producer=python3 -B scripts/lj104_sha_map_check.py {kit_arg} {repo_arg}",
        "script=scripts/lj104_sha_map.py",
        f"script_blob={blob.stdout.decode().strip()}",
        f"head={head.stdout.decode().strip()}",
    ]


def command_line(kit_arg: str, repo_arg: str) -> str:
    return (
        "command=python3 -B scripts/lj104_sha_map.py "
        f"--check {kit_arg} {repo_arg}"
    )


def write_stale(map_path: Path) -> None:
    lines = map_path.read_text().splitlines()
    found = 0
    out = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 5 and parts[4] == ROW:
            parts[0] = STALE
            parts[1] = STALE
            line = "\t".join(parts)
            found += 1
        out.append(line)
    if found != 1:
        fail("prediction row was not unique")
    map_path.write_text("\n".join(out) + "\n")


def run_check(kit: Path, repo: Path) -> tuple[str, int]:
    script = kit / "scripts" / "lj104_sha_map.py"
    result = subprocess.run(
        ["python3", "-B", str(script), "--check", str(kit), str(repo)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = result.stdout.decode()
    if text and not text.endswith("\n"):
        text += "\n"
    return text, result.returncode


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: lj104_sha_map_check.py <kit> <repo>",
            file=sys.stderr,
        )
        return 2
    kit_arg, repo_arg = sys.argv[1], sys.argv[2]
    require_relative(kit_arg)
    require_relative(repo_arg)
    kit = Path(kit_arg)
    repo = Path(repo_arg)
    map_path = kit / "logs" / "sha-map.tsv"
    saved = map_path.read_bytes()
    logs = kit / "logs"
    preface = identity(kit_arg, repo_arg, repo)
    invoked = command_line(kit_arg, repo_arg)
    try:
        write_stale(map_path)
        text, code = run_check(kit, repo)
        stale_body = preface + [
            "step=copy logs/sha-map.tsv",
            f"step=write {STALE} into both digest columns of {ROW}",
            invoked,
            text.rstrip("\n"),
            f"exit={code}",
        ]
        (logs / "sha-map-check-stale.txt").write_text(
            "\n".join(stale_body) + "\n"
        )
        if code == 0:
            fail("stale map was accepted")
        if map_path.read_bytes() == saved:
            fail("stale row was not written")
    finally:
        map_path.write_bytes(saved)
    if map_path.read_bytes() != saved:
        fail("map restore failed")
    text, code = run_check(kit, repo)
    current_body = preface + [
        "step=restore logs/sha-map.tsv",
        invoked,
        text.rstrip("\n"),
        f"exit={code}",
    ]
    (logs / "sha-map-check-current.txt").write_text(
        "\n".join(current_body) + "\n"
    )
    if code != 0:
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
