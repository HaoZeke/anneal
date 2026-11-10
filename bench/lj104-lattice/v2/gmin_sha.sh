#!/bin/bash
# Copyright (c) anneal developers
# Receipt of the GMIN executable, written before a scored run.
#
# Usage: gmin_sha.sh BINARY RECEIPT
# The receipt names the basename only, plus the sha256.
set -eu
binary=$1
receipt=$2
hash=$(sha256sum "$binary" | awk '{print $1}')
base=$(basename "$binary")
printf '%s  %s\n' "$hash" "$base" > "$receipt"
