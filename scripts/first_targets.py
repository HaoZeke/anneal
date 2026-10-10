#!/usr/bin/env python3
"""Print one first-target value per seed from a hist campaign arm, for
first_passage_fit: the forces at which the ensemble first reached the
target, or `-` for a seed that did not.

Usage: first_targets.py <campaign-dir> <arm-name>
"""
import glob
import os
import re
import sys

campaign, arm = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(campaign, f"{arm}_*.out")), key=lambda f: int(re.search(r"_(\d+)\.out$", f).group(1)))
for path in files:
    text = open(path, errors="replace").read()
    m = re.search(r"first_target_calls (\d+)", text)
    print(m.group(1) if m else "-")
