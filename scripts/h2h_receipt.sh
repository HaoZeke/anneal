#!/bin/sh
# Copyright (c) anneal developers
# Record one or more commands with the head, the tree, and both clocks.
# Every arm in one invocation shares a private mount namespace and a
# private UTS namespace. uname -n and the hostname file both read bench.
# The host copy of that file is left untouched.
# The receipt is the raw text. Nothing is rewritten after the run.
#
# One arm:
#   sh scripts/h2h_receipt.sh <log> -- <command>
# Several arms, one namespace, one binary hash taken before the first arm:
#   sh scripts/h2h_receipt.sh --batch
# Each batch line is a log path, a tab, and the command.

set -eu

if [ "${1:-}" != "--inside" ]; then
  exec unshare --user --map-root-user --mount --uts --fork \
    sh "$0" --inside "$@"
fi
shift

printf 'bench\n' >h2h-hostname
mount --bind "$PWD/h2h-hostname" /etc/hostname
hostname bench
trap 'rm -f h2h-hostname' EXIT

hn=$(tr -d '\n' </etc/hostname)
nn=$(uname -n)
if [ "$hn" != "bench" ] || [ "$nn" != "bench" ]; then
  echo "bench namespace is not in place" >&2
  exit 1
fi

hash_file() {
  sha256sum "$1" | awk 'NR==1 { print $1 }'
}

# Compiler, standard library, and the shared libraries the binary links.
# The receipt prints basenames and digests, not directories.
package_lines() {
  bin=$1
  if command -v rustc >/dev/null 2>&1; then
    sysroot=$(rustc --print sysroot)
    if [ -f "$sysroot/bin/rustc" ]; then
      echo "package_rustc=$(hash_file "$sysroot/bin/rustc")"
    fi
    if [ -f "$sysroot/bin/cargo" ]; then
      echo "package_cargo=$(hash_file "$sysroot/bin/cargo")"
    fi
    for lib in "$sysroot"/lib/rustlib/*/lib/libstd-*.rlib; do
      if [ -f "$lib" ]; then
        base=$(basename "$lib")
        echo "package_${base}=$(hash_file "$lib")"
      fi
    done
  fi
  if command -v ldd >/dev/null 2>&1 && [ -n "$bin" ] && [ -f "$bin" ]; then
    ldd "$bin" 2>/dev/null | while IFS= read -r line; do
      path=$(printf '%s\n' "$line" | sed -n 's/.*=> *\(\/[^ ]*\).*/\1/p')
      if [ -z "$path" ]; then
        path=$(printf '%s\n' "$line" | sed -n 's/^\(\/[^ ]*\).*/\1/p')
      fi
      if [ -n "$path" ] && [ -f "$path" ]; then
        base=$(basename "$path")
        echo "package_${base}=$(hash_file "$path")"
      fi
    done
  fi
}

first_binary() {
  printf '%s\n' "$1" | tr '\t' ' ' | tr -s ' ' '\n' | while IFS= read -r word; do
    case $word in
      *=*) continue ;;
    esac
    if [ -f "$word" ] && [ -x "$word" ]; then
      printf '%s\n' "$word"
      break
    fi
  done
}

BIN_SHA=""
PACKAGES=""

prepare_binary() {
  cmd=$1
  if [ -n "$BIN_SHA" ]; then
    return 0
  fi
  found=$(first_binary "$cmd" || true)
  if [ -z "$found" ]; then
    echo "no binary to hash before the run" >&2
    exit 1
  fi
  BIN_SHA=$(hash_file "$found")
  PACKAGES=$(package_lines "$found" | sort -u)
}

run_arm() {
  log=$1
  cmd=$2
  prepare_binary "$cmd"
  head_sha=$(git rev-parse HEAD)
  tree_sha=$(git rev-parse 'HEAD^{tree}')
  pins=$(printf '%s' "$cmd" | sed -n 's/.*taskset -c \([^ ]*\).*/\1/p')
  mkdir -p "$(dirname "$log")"
  {
    echo "command=$cmd"
    echo "head=$head_sha"
    echo "tree=$tree_sha"
    echo "pins=${pins:-unset}"
    echo "uname_n=$(uname -n)"
    echo "hostname_sha256=$(hash_file /etc/hostname)"
    printf '%s\n' "$PACKAGES"
    echo "binary_sha256=$BIN_SHA"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime_start=$(cut -d' ' -f1 /proc/uptime)"
    echo "nproc=$(nproc)"
    echo "mhz=$(awk -F': ' '/cpu MHz/{print $2; exit}' /proc/cpuinfo)"
    lscpu
    echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "start_cst=$(TZ=America/Chicago date '+%Y-%m-%d %H:%M:%S CST')"
  } >"$log"
  set +e
  sh -c "$cmd" >>"$log" 2>&1
  status=$?
  set -e
  {
    echo "exit=$status"
    echo "boot_id_end=$(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime_end=$(cut -d' ' -f1 /proc/uptime)"
    echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "end_cst=$(TZ=America/Chicago date '+%Y-%m-%d %H:%M:%S CST')"
  } >>"$log"
  return "$status"
}

if [ "${1:-}" = "--batch" ]; then
  fail=0
  tab=$(printf '\t')
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in
      '' | '#'*) continue ;;
    esac
    log=${line%%"$tab"*}
    cmd=${line#*"$tab"}
    if [ "$log" = "$line" ] || [ -z "$cmd" ]; then
      echo "batch line needs a tab between the log and the command" >&2
      exit 2
    fi
    run_arm "$log" "$cmd" || fail=1
  done
  exit "$fail"
fi

log=${1:-}
shift || true
if [ "${1:-}" != "--" ] || [ -z "$log" ]; then
  echo "usage: sh scripts/h2h_receipt.sh <log> -- <command>" >&2
  echo "       sh scripts/h2h_receipt.sh --batch" >&2
  exit 2
fi
shift
run_arm "$log" "$*"
exit $?
