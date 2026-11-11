#!/bin/bash
# Copyright (c) anneal developers
# Enter the bench UTS namespace, print hostname from inside it, then run AGOX.
set -eu
cd /work/anneal
exec sudo unshare --uts sh -c 'hostname bench; hostname; exec sudo -u ubuntu /work/anneal/.venv/bin/python3 -u /work/anneal/scripts/lj38_paired/agox_documented.py'
