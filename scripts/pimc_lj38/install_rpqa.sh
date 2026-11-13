#!/usr/bin/env bash
# Pin Fiorentino's public RPQA tree. The LJ example and the rpqa smotion
# are on feature/rpqa-reset-momenta at this commit.
set -euo pipefail

WORK=/work/pimc-lj38
PIN=4e86f5fdc1583ef21f105fc2bed96a279b2430c1
URL=https://github.com/afiorent/i-pi.git
if [[ ! -d /work ]]; then
  sudo -n mkdir -p /work
  sudo -n chown "$(id -u):$(id -g)" /work
fi
mkdir -p "$WORK"
DEST=$WORK/i-pi
if [[ ! -d $DEST/.git ]]; then
  git clone --branch feature/rpqa-reset-momenta "$URL" "$DEST"
fi
git -C "$DEST" fetch origin feature/rpqa-reset-momenta
if ! git -C "$DEST" cat-file -e "$PIN^{commit}" 2>/dev/null; then
  echo "pin $PIN is not in $DEST" >&2
  exit 1
fi
git -C "$DEST" checkout --detach "$PIN"
test -f "$DEST/examples/features/quantum_annealing/RPQA/LJ_cluster/input.xml"
test -f "$DEST/ipi/engine/smotion/rpqa.py"
cat > "$WORK/rpqa-source.txt" << EOF
rpqa_commit=$PIN
rpqa_source=$URL
rpqa_branch=feature/rpqa-reset-momenta
rpqa_example=examples/features/quantum_annealing/RPQA/LJ_cluster
EOF
echo "rpqa pin=$PIN"
