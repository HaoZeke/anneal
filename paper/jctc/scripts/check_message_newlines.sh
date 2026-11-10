#!/bin/sh
# Copyright (c) anneal developers
# Fail when a commit message ends in more than one newline.
# Usage: check_message_newlines.sh [BASE]
set -eu
base=${1:-3675c9c375bbe560773cdf860bc672b8cbfe74d2}
git rev-list --reverse --first-parent "$base"..HEAD | python3 -c '
import subprocess
import sys
bad = 0
for sha in sys.stdin.read().split():
    raw = subprocess.check_output(["git", "cat-file", "commit", sha])
    message = raw.split(b"\n\n", 1)[1]
    extra = len(message) - len(message.rstrip(b"\n"))
    if extra != 1:
        print(f"{sha} trailing_newlines {extra}")
        bad = 1
if bad:
    sys.exit(1)
print("messages end in one newline")
'
