# Record one command with the head, the tree, and both clocks.
# Usage: sh scripts/h2h_receipt.sh <log> -- <command>
log=$1
shift
if [ "$1" != "--" ]; then
  echo "usage: sh scripts/h2h_receipt.sh <log> -- <command>" >&2
  exit 2
fi
shift
head_sha=$(git rev-parse HEAD)
tree_sha=$(git rev-parse 'HEAD^{tree}')
pins=$(printf '%s' "$*" | sed -n 's/.*taskset -c \([^ ]*\).*/\1/p')
{
  echo "command=$*"
  echo "head=$head_sha"
  echo "tree=$tree_sha"
  echo "pins=${pins:-unset}"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "uptime_start=$(cut -d' ' -f1 /proc/uptime)"
  echo "nproc=$(nproc)"
  echo "mhz=$(awk -F': ' '/cpu MHz/{print $2; exit}' /proc/cpuinfo)"
  lscpu
  echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "start_cst=$(TZ=America/Chicago date '+%Y-%m-%d %H:%M:%S CST')"
} >"$log"
set +e
"$@" >>"$log" 2>&1
status=$?
set -e
{
  echo "exit=$status"
  echo "boot_id_end=$(cat /proc/sys/kernel/random/boot_id)"
  echo "uptime_end=$(cut -d' ' -f1 /proc/uptime)"
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "end_cst=$(TZ=America/Chicago date '+%Y-%m-%d %H:%M:%S CST')"
} >>"$log"
exit "$status"
