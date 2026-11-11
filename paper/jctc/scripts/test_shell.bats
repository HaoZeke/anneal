#!/usr/bin/env bats
# Copyright (c) anneal developers
# Behaviour tests for the shell scripts in the coverage list.

setup() {
  ROOT="$(cd "$BATS_TEST_DIRNAME/../../.." && pwd)"
  WORK="$BATS_TEST_TMPDIR/work"
  mkdir -p "$WORK"
}

@test "message check accepts one trailing newline and rejects two" {
  git -C "$WORK" init
  git -C "$WORK" config user.email "rgoswami@ieee.org"
  git -C "$WORK" config user.name "Rohit Goswami"
  printf 'x\n' >"$WORK/f"
  git -C "$WORK" add f
  git -C "$WORK" commit -m "fine"
  base="$(git -C "$WORK" rev-parse HEAD)"
  printf 'y\n' >"$WORK/f"
  git -C "$WORK" add f
  tree="$(git -C "$WORK" write-tree)"
  new="$(printf 'two\n\n' | git -C "$WORK" -c commit.gpgsign=false commit-tree "$tree" -p "$base")"
  git -C "$WORK" reset --soft "$new"
  cd "$WORK"
  run sh "$ROOT/paper/jctc/scripts/check_message_newlines.sh" "$base"
  [ "$status" -eq 1 ]
  [[ "$output" == *"trailing_newlines"* ]]
}

@test "pack script refuses a tree whose commit is absent" {
  git -C "$WORK" init
  git -C "$WORK" config user.email "rgoswami@ieee.org"
  git -C "$WORK" config user.name "Rohit Goswami"
  printf 'x\n' >"$WORK/Cargo.toml"
  git -C "$WORK" add Cargo.toml
  git -C "$WORK" commit -m "init"
  tree="$(git -C "$WORK" rev-parse 'HEAD^{tree}')"
  cd "$WORK"
  run sh "$ROOT/paper/jctc/scripts/pack_lj75_source.sh" "$tree" "$WORK/out.tar.xz"
  [ "$status" -eq 1 ]
  [[ "$output" == *"not present"* ]]
}

@test "hardware capture records the live host" {
  sleep 30 &
  pid=$!
  export BUILD_HEAD=6146da08dbab883c21a5607bfa1da7d4677a8008
  export BUILD_TREE=37535ec1a6e99d47a2afe0752072a33af64dccb3
  run sh "$ROOT/paper/jctc/scripts/lj75_hardware_capture.sh" "$pid" "$WORK/receipt.txt"
  kill "$pid"
  [ "$status" -eq 0 ]
  grep -q '^host ' "$WORK/receipt.txt"
  ! grep -q 'host-a' "$WORK/receipt.txt"
}

@test "copper build rejects the wrong rgpot commit" {
  git -C "$WORK" init
  git -C "$WORK" config user.email "rgoswami@ieee.org"
  git -C "$WORK" config user.name "Rohit Goswami"
  printf 'x\n' >"$WORK/f"
  git -C "$WORK" add f
  git -C "$WORK" commit -m "init"
  run bash "$ROOT/scripts/build_librgpot_cuh2.sh" "$WORK" "$WORK/out.so"
  [ "$status" -eq 2 ]
  [[ "$output" == *"is not"* ]]
}

@test "copper driver stops when the library is missing" {
  printf 'rust-version = "1.88"\n' >"$WORK/Cargo.toml"
  mkdir -p "$WORK/scripts"
  cp "$ROOT/scripts/run_cuh2_fcc_rec.sh" "$WORK/scripts/"
  run bash "$WORK/scripts/run_cuh2_fcc_rec.sh"
  [ "$status" -eq 1 ]
  [[ "$output" == *"missing"* ]]
}
