#!/usr/bin/env bash
# Sync this checkout's sources to the Elja staging tree and stamp it.
#
# Usage: elja_sync_staging.sh [host=elja] [staging=~/anneal-psym]
#
# Copies src/, examples/, tests/, scripts/, Cargo.toml and Cargo.lock, and
# writes SOURCE_COMMIT (the local HEAD, suffixed -dirty when the tree has
# uncommitted changes to those paths) so a build's provenance names the
# revision it came from. The staging tree is not a git checkout.
set -euo pipefail
HOST=${1:-elja}
STAGING=${2:-'~/anneal-psym'}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
commit=$(git rev-parse HEAD)
if ! git diff --quiet HEAD -- src examples tests scripts Cargo.toml Cargo.lock; then
  commit="${commit}-dirty"
fi
for dir in src examples tests scripts vendor; do
  rsync -a --delete "$dir/" "$HOST:$STAGING/$dir/"
done
rsync -a Cargo.toml Cargo.lock "$HOST:$STAGING/"
ssh "$HOST" "printf '%s\n' '$commit' > $STAGING/SOURCE_COMMIT"
echo "staged $commit at $HOST:$STAGING"
