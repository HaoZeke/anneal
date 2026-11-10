# Copyright (c) anneal developers
"""One commit-tree pass. No sleeps. No author or committer dates are set."""
import os
import subprocess
from pathlib import Path

REPO = Path(".")
MSGDIR = Path(os.environ["REPLAY_MSGDIR"])
MAIN = "3675c9c375bbe560773cdf860bc672b8cbfe74d2"
DEVELOP = "91b15a024749320b9463e4abbdf10d37e933174b"
OLDTIP = "9109df21fbfd9470c16cff37a7a7b5228bb809e0"


def git(*args, env=None, data=None):
    proc = subprocess.run(
        ["git", "-C", str(REPO), *args],
        input=data,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode()[-400:])
    return proc.stdout


def main():
    rows = git(
        "log", "--first-parent", "--reverse", "--format=%H %T",
        f"{MAIN}..{OLDTIP}",
    ).decode().splitlines()
    base = os.environ.copy()
    for key in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"):
        base.pop(key, None)
    base.update({
        "TZ": "America/Chicago",
        "GIT_AUTHOR_NAME": "Rohit Goswami",
        "GIT_AUTHOR_EMAIL": "rgoswami@ieee.org",
        "GIT_COMMITTER_NAME": "Rohit Goswami",
        "GIT_COMMITTER_EMAIL": "rgoswami@ieee.org",
    })
    parent = MAIN
    made = []
    for index, line in enumerate(rows, 1):
        old, tree = line.split()
        path = MSGDIR / f"{index:04d}.txt"
        if path.exists():
            body = path.read_bytes()
        else:
            raw = git("cat-file", "commit", old)
            body = raw.split(b"\n\n", 1)[1]
        if not body.endswith(b"\n") or body.endswith(b"\n\n"):
            raise SystemExit(f"message {index} newline")
        parents = ["-p", parent]
        if index == 1:
            parents += ["-p", DEVELOP]
        sha = git(
            "-c", "commit.gpgsign=false", "commit-tree", tree, *parents,
            env=base, data=body,
        ).decode().strip()
        raw = git("cat-file", "commit", sha)
        stamp = None
        for entry in raw.splitlines():
            if entry.startswith(b"author "):
                stamp = entry.decode().rsplit(" ", 2)
        made.append((index, old, sha, tree, int(stamp[-2]), stamp[-1]))
        parent = sha
    earlier = [
        made[i][0] for i in range(1, len(made)) if made[i][4] < made[i - 1][4]
    ]
    shared = [
        made[i][0] for i in range(1, len(made)) if made[i][4] == made[i - 1][4]
    ]
    print("earlier", earlier)
    print("shared_with_parent", shared)
    print("first", made[0][2], made[0][4], made[0][5])
    print("last", made[-1][2], made[-1][4], made[-1][5])
    print("distinct_seconds", len({row[4] for row in made}), "commits", len(made))
    print("TIP48", made[-1][2])
    for index, old, sha, tree, _ts, _tz in made:
        print(f"MAP {index:02d} {old} {sha} {tree}")


if __name__ == "__main__":
    main()
